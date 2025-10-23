from .optimal_transport import OptimalTransport, WeightAwareOptimalTransport, EnhancedWeightAwareOptimalTransport
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

from .data import get_loaders 
from .model import rm_modules
from tqdm import tqdm


def check_sparsity(model):#, save_path):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()

    model.config.use_cache = use_cache
    return float(count)

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
    
def apply_intra_fusion_to_mlp(layer, ot_map, pruning_idxs, device):
    """MLPレイヤーにIntra-Fusionを適用"""
    
    def apply_optimal_transport_to_weight(weight, ot_map, dim):
        """重みに最適輸送マップを適用"""
        if dim == 0:  # 出力次元（gate_proj, up_proj）
            # [intermediate_size, input_size] -> [fused_size, input_size]
            fused_weight = torch.einsum('ij,jk->ik', ot_map, weight)
        elif dim == 1:  # 入力次元（down_proj）
            # [output_size, intermediate_size] -> [output_size, fused_size]  
            fused_weight = torch.einsum('ij,kj->ki', ot_map, weight)
        
        return fused_weight
    
    # 残存するチャネル数
    keep_idxs = list(set(range(ot_map.shape[1])) - set(pruning_idxs.tolist()))
    new_intermediate_size = len(keep_idxs)
    
    with torch.no_grad():
        # gate_projの重み融合
        gate_weight = layer.mlp.gate_proj.weight.data
        fused_gate_weight = apply_optimal_transport_to_weight(gate_weight, ot_map, 0)
        
        # 新しいLinear層の作成
        new_gate_proj = nn.Linear(
            layer.mlp.gate_proj.in_features, 
            new_intermediate_size, 
            bias=layer.mlp.gate_proj.bias is not None
        ).to(device)
        new_gate_proj.weight.data = fused_gate_weight
        if layer.mlp.gate_proj.bias is not None:
            new_gate_proj.bias.data = torch.einsum('ij,j->i', ot_map, layer.mlp.gate_proj.bias.data)
        
        # up_projも同様に処理
        up_weight = layer.mlp.up_proj.weight.data
        fused_up_weight = apply_optimal_transport_to_weight(up_weight, ot_map, 0)
        
        new_up_proj = nn.Linear(
            layer.mlp.up_proj.in_features,
            new_intermediate_size,
            bias=layer.mlp.up_proj.bias is not None
        ).to(device)
        new_up_proj.weight.data = fused_up_weight
        if layer.mlp.up_proj.bias is not None:
            new_up_proj.bias.data = torch.einsum('ij,j->i', ot_map, layer.mlp.up_proj.bias.data)
        
        # down_projは入力次元を変更
        down_weight = layer.mlp.down_proj.weight.data
        fused_down_weight = apply_optimal_transport_to_weight(down_weight, ot_map, 1)
        
        new_down_proj = nn.Linear(
            new_intermediate_size,
            layer.mlp.down_proj.out_features,
            bias=layer.mlp.down_proj.bias is not None
        ).to(device)
        new_down_proj.weight.data = fused_down_weight
        if layer.mlp.down_proj.bias is not None:
            new_down_proj.bias.data = layer.mlp.down_proj.bias.data.clone()
        
        # レイヤーの置き換え
        layer.mlp.gate_proj = new_gate_proj
        layer.mlp.up_proj = new_up_proj  
        layer.mlp.down_proj = new_down_proj
        
        # intermediate_sizeの更新
        layer.mlp.intermediate_size = new_intermediate_size

    print(f"Layer compressed: {gate_weight.shape[0]} -> {new_intermediate_size} channels")
    
def create_mlp_group_for_optimal_transport(layer_idx, model, device):
    """MLP層用の簡易グループ構造を作成"""
    layer = model.model.layers[layer_idx]
    
    # MLP内の3つのプロジェクション層
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    down_proj = layer.mlp.down_proj
    
    # 中間次元のサイズ
    intermediate_size = gate_proj.out_features
    all_idxs = list(range(intermediate_size))
    
    class SimpleDep:
        def __init__(self, module, is_down_proj=False):
            self.target = SimpleTarget(module)
            self.handler = "prune_linear_in_channels" if is_down_proj else "prune_linear_out_channels"
    
    class SimpleTarget:
        def __init__(self, module):
            self.module = module
    
    # グループ構造の作成
    group = [
        (SimpleDep(gate_proj, False), all_idxs),    # gate_projの出力次元
        (SimpleDep(up_proj, False), all_idxs),      # up_projの出力次元  
        (SimpleDep(down_proj, True), all_idxs),     # down_projの入力次元
    ]
    
    return group

def calculate_weight_level_importance(layer_idx, rm_weights, fo_grads, snip_grads):
    """重み単位の重要度スコアを計算"""
    
    # 各プロジェクション層の重み単位スコア
    W_down_fo = rm_weights[layer_idx+64] * fo_grads[layer_idx+64]
    W_up_fo = (rm_weights[layer_idx+32] * fo_grads[layer_idx+32]).t()
    W_gate_fo = (rm_weights[layer_idx] * fo_grads[layer_idx]).t()
    
    W_down_snip = rm_weights[layer_idx+64] * snip_grads[layer_idx+64]
    W_up_snip = (rm_weights[layer_idx+32] * snip_grads[layer_idx+32]).t()
    W_gate_snip = (rm_weights[layer_idx] * snip_grads[layer_idx]).t()
    
    # 重み単位のFOとSNIPスコア（標準化前）
    weight_fo_scores = torch.stack([W_down_fo, W_up_fo, W_gate_fo], dim=0)  # [3, output_dim, input_dim]
    weight_snip_scores = torch.stack([W_down_snip, W_up_snip, W_gate_snip], dim=0)
    
    return weight_fo_scores, weight_snip_scores

def compute_channel_importance_statistics(weight_fo_scores, weight_snip_scores):
    """チャネル単位の重要度統計を計算"""
    
    def compute_mutual_information_importance(weight_scores):
        """相互情報量ベースの重要度計算"""
        # 簡易版：チャネル間の相関を重要度とする
        
        # weight_scores: [3, output_dim, intermediate_size]
        flattened = weight_scores.flatten(0, 1)  # [3*output_dim, intermediate_size]
        
        # 各チャネルと他のチャネルとの平均相関
        correlation_matrix = torch.corrcoef(flattened.t())  # [intermediate_size, intermediate_size]
        
        # 対角要素を除いた平均相関
        mask = ~torch.eye(correlation_matrix.shape[0], dtype=bool)
        mutual_info = correlation_matrix.abs()[mask].reshape(correlation_matrix.shape[0], -1).mean(dim=1)
        
        return mutual_info
    
    def compute_structural_importance(weight_scores):
        """構造的重要度を計算（特異値分解ベース）"""
        # weight_scores: [3, output_dim, intermediate_size]
        
        structural_scores = []
        for i in range(weight_scores.shape[2]):  # 各チャネルについて
            channel_weights = weight_scores[:, :, i]  # [3, output_dim]
            
            # SVDによる構造分析
            try:
                U, S, V = torch.svd(channel_weights)
                # 主特異値の大きさを重要度とする
                structural_importance = S[0] if len(S) > 0 else torch.tensor(0.0)
            except:
                structural_importance = torch.tensor(0.0)
            
            structural_scores.append(structural_importance)
        
        return torch.stack(structural_scores)

    # チャネル次元での統計量計算
    channel_stats = {}
    
    # 平均（従来手法）
    fo_mean = weight_fo_scores.mean(dim=(0, 1))  # [intermediate_size]
    snip_mean = weight_snip_scores.mean(dim=(0, 1))
    
    # 最大値（重要な重みを保護）
    fo_max = weight_fo_scores.max(dim=1)[0].max(dim=0)[0]
    snip_max = weight_snip_scores.max(dim=1)[0].max(dim=0)[0]
    
    # 標準偏差（多様性を考慮）
    fo_std = weight_fo_scores.std(dim=(0, 1))
    snip_std = weight_snip_scores.std(dim=(0, 1))
    
    # 1. 上位パーセンタイルスコア（外れ値に頑健）
    fo_p95 = torch.quantile(weight_fo_scores.flatten(0, 1), 0.95, dim=0)
    snip_p95 = torch.quantile(weight_snip_scores.flatten(0, 1), 0.95, dim=0)
    
    # 2. 重要度の一貫性（3つの層間での類似性）
    fo_consistency = -weight_fo_scores.std(dim=0).mean(dim=0)  # 低い標準偏差 = 高い一貫性
    snip_consistency = -weight_snip_scores.std(dim=0).mean(dim=0)
    
    # 3. 重み行列の構造的重要度（特異値分解ベース）
    fo_structural = compute_structural_importance(weight_fo_scores)
    snip_structural = compute_structural_importance(weight_snip_scores)
    
    # 4. 相互情報量ベースの重要度
    fo_mutual_info = compute_mutual_information_importance(weight_fo_scores)
    snip_mutual_info = compute_mutual_information_importance(weight_snip_scores)
    
    # 重要な重みの割合（閾値以上の重みの比率）
    fo_threshold = weight_fo_scores.mean() + weight_fo_scores.std()
    snip_threshold = weight_snip_scores.mean() + weight_snip_scores.std()
    fo_important_ratio = (weight_fo_scores > fo_threshold).float().mean(dim=(0, 1))
    snip_important_ratio = (weight_snip_scores > snip_threshold).float().mean(dim=(0, 1))
    
    channel_stats = {
        'fo_mean': fo_mean,'snip_mean': snip_mean,
        'fo_max': fo_max,'snip_max': snip_max,
        'fo_std': fo_std,'snip_std': snip_std,
        'fo_p95': fo_p95, 'snip_p95': snip_p95,
        'fo_consistency': fo_consistency, 'snip_consistency': snip_consistency,
        'fo_structural': fo_structural, 'snip_structural': snip_structural,
        'fo_mutual_info': fo_mutual_info, 'snip_mutual_info': snip_mutual_info,
        'fo_important_ratio': fo_important_ratio,'snip_important_ratio': snip_important_ratio
    }
    
    return channel_stats

def Structured_AFR_with_IntraFusion(args, model, tokenizer, device):
    
    # 最適輸送オブジェクトの初期化
    ot = OptimalTransport(
        p=1,                           # L1ノルム距離
        target_probability="uniform",   # または "importance"  
        source_probability="importance",   # または "importance"
        target="most_important",       # または "cluster_centroid"
        gpu_id=device if isinstance(device, int) else 0
    )
    
    # 既存のFO + SNIP重要度計算（変更なし）
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("loading calibration data")
    dataloader, _ = get_loaders(nsamples=args.nsamples, seed=args.seed, 
                               seqlen=model.seqlen, tokenizer=tokenizer)
    
    # [既存のFO計算部分 - 変更なし]
    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    
    def store_feature(module, input, output):
        # 既存のコードと同じ
        global P_SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        output = output.reshape(output.size(0), -1)
        U, S, Vh = torch.svd(output)
        singular_value_mean = S.mean()
        P_SVD_loss = P_SVD_loss + singular_value_mean

    # フック登録とFO勾配計算
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook)

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    # FO勾配計算
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        break
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))
    
    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)

    # SNIP勾配計算  
    outputs = model(inputs)
    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    snip_grads = list(torch.autograd.grad(loss, rm_weights))

    # 各層でスコア計算と最適輸送適用
    layers = model.model.layers
    
    for layer_idx in tqdm(range(len(layers)), desc="Processing layers with Intra-Fusion"):
        # 既存のスコア計算
        W_down_fo = rm_weights[layer_idx+64] * fo_grads[layer_idx+64]
        W_up_fo = (rm_weights[layer_idx+32] * fo_grads[layer_idx+32]).t()
        W_gate_fo = (rm_weights[layer_idx] * fo_grads[layer_idx]).t()
        fo_metric = (W_down_fo + W_up_fo + W_gate_fo).mean(axis=0)
        
        W_down_snip = rm_weights[layer_idx+64] * snip_grads[layer_idx+64]
        W_up_snip = (rm_weights[layer_idx+32] * snip_grads[layer_idx+32]).t()
        W_gate_snip = (rm_weights[layer_idx] * snip_grads[layer_idx]).t()
        snip_metric = (W_down_snip + W_up_snip + W_gate_snip).mean(axis=0)
        
        # スコアの標準化と結合
        fo_standardized = (fo_metric - fo_metric.mean()) / fo_metric.std()
        snip_standardized = (snip_metric - snip_metric.mean()) / snip_metric.std()
        combined_score = fo_standardized + snip_standardized
        
        # プルーニング対象の決定
        n_prune = int(combined_score.shape[0] * args.pruning_ratio)
        _, pruning_idxs = torch.topk(combined_score, n_prune, largest=False)
        
        # 最適輸送マップの計算
        group = create_mlp_group_for_optimal_transport(layer_idx, model, device)
        ot_map = ot(group, combined_score, pruning_idxs)
        
        # Intra-Fusionを適用したプルーニング
        apply_intra_fusion_to_mlp(layers[layer_idx], ot_map, pruning_idxs, device)

    model.config.use_cache = use_cache

def enhanced_structured_afr_with_intrafusion(args, model, tokenizer, device):
    """改良版のStructured AFR with IntraFusion"""
    
    # ... 既存の初期化とFO/SNIP勾配計算部分は同じ ...
    # 既存のFO + SNIP重要度計算（変更なし）
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("loading calibration data")
    dataloader, _ = get_loaders(nsamples=args.nsamples, seed=args.seed, 
                               seqlen=model.seqlen, tokenizer=tokenizer)
    
    # [既存のFO計算部分 - 変更なし]
    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    
    def store_feature(module, input, output):
        # 既存のコードと同じ
        global P_SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        output = output.reshape(output.size(0), -1)
        U, S, Vh = torch.svd(output)
        singular_value_mean = S.mean()
        P_SVD_loss = P_SVD_loss + singular_value_mean

    # フック登録とFO勾配計算
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook)

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    # FO勾配計算
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        break
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))
    
    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)

    # SNIP勾配計算  
    outputs = model(inputs)
    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    snip_grads = list(torch.autograd.grad(loss, rm_weights))

    # 各層でスコア計算と最適輸送適用
    layers = model.model.layers

    # 重み対応最適輸送の初期化
    original_ot = OptimalTransport(
        p=1,
        target_probability="uniform", # or weight_aware or max_preserving or diversity_aware
        source_probability="weight_aware", # or importance
        target="most_important",
        gpu_id=device if isinstance(device, int) else 0
    )
    weight_aware_ot = WeightAwareOptimalTransport(original_ot)
    
    layers = model.model.layers
    
    for layer_idx in tqdm(range(len(layers)), desc="Processing layers with Weight-Aware Intra-Fusion"):
        # 重み単位の重要度スコア計算
        weight_fo_scores, weight_snip_scores = calculate_weight_level_importance(
            layer_idx, rm_weights, fo_grads, snip_grads
        )
        
        # チャネル単位統計計算
        channel_stats = compute_channel_importance_statistics(weight_fo_scores, weight_snip_scores)
        
        # 従来の集約スコア（プルーニング対象決定用）
        fo_metric = weight_fo_scores.mean(dim=(0, 1))
        snip_metric = weight_snip_scores.mean(dim=(0, 1))
        fo_standardized = (fo_metric - fo_metric.mean()) / fo_metric.std()
        snip_standardized = (snip_metric - snip_metric.mean()) / snip_metric.std()
        combined_score = fo_standardized + snip_standardized
        
        # プルーニング対象決定
        n_prune = int(combined_score.shape[0] * args.pruning_ratio)
        _, pruning_idxs = torch.topk(combined_score, n_prune, largest=False)
        
        # 重み単位統計を考慮した最適輸送マップ計算
        group = create_mlp_group_for_optimal_transport(layer_idx, model, device)
        ot_map = weight_aware_ot(group, combined_score, pruning_idxs, channel_stats, layer_idx)
        
        # Intra-Fusionを適用
        apply_intra_fusion_to_mlp(layers[layer_idx], ot_map, pruning_idxs, device)
    # 最終サマリー出力
    print("\nGenerating final rescue analysis report...")
    final_summary = weight_aware_ot.get_rescue_summary()

    model.config.use_cache = use_cache
    
    
def enhanced_structured_afr_v2(args, model, tokenizer, device):
    """性能改善版のStructured AFR"""
    
    # 複数の戦略を試行
    strategies = [
        ("comprehensive", "entropy_regularized", "adaptive"),
        ("conservative", "robust", "adaptive"), 
        ("balanced", "adaptive", "robust")
    ]
    
    best_strategy = strategies[0] if not hasattr(args, 'strategy_idx') else strategies[args.strategy_idx]
    
    original_ot = OptimalTransport(
        p=1, 
        target_probability="uniform", 
        source_probability="importance",
        target="most_important",
        gpu_id=device if isinstance(device, int) else 0
    )
    
    enhanced_ot = EnhancedWeightAwareOptimalTransport(
        original_ot, 
        importance_strategy=best_strategy[0],
        source_prob_type=best_strategy[1],
        target_prob_type=best_strategy[2]
    )
    
    # 既存の初期化処理...
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("loading calibration data")
    dataloader, _ = get_loaders(nsamples=args.nsamples, seed=args.seed, 
                               seqlen=model.seqlen, tokenizer=tokenizer)
    
    # FO + SNIP計算（既存と同じ）
        # [既存のFO計算部分 - 変更なし]
    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    
    def store_feature(module, input, output):
        # 既存のコードと同じ
        global P_SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        output = output.reshape(output.size(0), -1)
        U, S, Vh = torch.svd(output)
        singular_value_mean = S.mean()
        P_SVD_loss = P_SVD_loss + singular_value_mean

    # フック登録とFO勾配計算
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook)

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    # FO勾配計算
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        break
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))
    
    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)

    # SNIP勾配計算  
    outputs = model(inputs)
    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    snip_grads = list(torch.autograd.grad(loss, rm_weights))
    
    layers = model.model.layers
    
    for layer_idx in tqdm(range(len(layers)), desc=f"Enhanced pruning with {best_strategy}"):
        # 改良された重み重要度計算
        weight_fo_scores, weight_snip_scores = calculate_weight_level_importance(
            layer_idx, rm_weights, fo_grads, snip_grads
        )
        
        # 高度なチャネル統計
        channel_stats = compute_channel_importance_statistics(weight_fo_scores, weight_snip_scores)
        
        # プルーニング対象決定（従来通り）
        fo_metric = weight_fo_scores.mean(dim=(0, 1))
        snip_metric = weight_snip_scores.mean(dim=(0, 1))
        fo_standardized = (fo_metric - fo_metric.mean()) / (fo_metric.std() + 1e-8)
        snip_standardized = (snip_metric - snip_metric.mean()) / (snip_metric.std() + 1e-8)
        combined_score = fo_standardized + snip_standardized
        

        n_prune = int(combined_score.shape[0] * args.pruning_ratio)
        _, pruning_idxs = torch.topk(combined_score, n_prune, largest=False)
        
        # 改良された最適輸送
        group = create_mlp_group_for_optimal_transport(layer_idx, model, device)
        ot_map = enhanced_ot(group, combined_score, pruning_idxs, channel_stats)
        
        # Intra-Fusion適用
        apply_intra_fusion_to_mlp(layers[layer_idx], ot_map, pruning_idxs, device)

    model.config.use_cache = use_cache
    