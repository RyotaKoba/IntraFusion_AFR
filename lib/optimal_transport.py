import torch
import torch.nn as nn
import numpy as np
import ot
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict

import typing
from .pruner import function
from sklearn.mixture import GaussianMixture


class OptimalTransport:
    """
    OptimalTransport class for computing the transport map matching similar neural pairings

    Args:
        p (int):  p value for the p-norm distance for calculating cost between neural pairings.
        target_probability (str): Target probability of the Optimal Transport problem.
        source_probability (str): Source probability of the Optimal Transport problem.
        target (str): Target used for the Optimal Transport problem. Either "most_important" or "cluster_centroid".
        gpu_id (int): GPU ID of the GPU used. Use "-1" for CPU. 
    """

    def __init__(
        self,
        p: int = 1,
        target_probability: str = "uniform",
        source_probability: str = "uniform",
        target: str = "most_important",
        gpu_id: int = 0,
    ):
        self.p = p
        self.target_probability = target_probability
        self.source_probability = source_probability
        self.target = target
        self.gpu_id = gpu_id

    def _normalize(self, cost, normalizer):
        if normalizer is None:
            return cost
        elif isinstance(normalizer, typing.Callable):
            return normalizer(cost)
        elif normalizer == "sum":
            return cost / cost.sum()
        elif normalizer == "standardization":
            return (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
        elif normalizer == "mean":
            return cost / cost.mean()
        elif normalizer == "max":
            return cost / cost.max()
        elif normalizer == "gaussian":
            return (cost - cost.mean()) / (cost.std() + 1e-8)
        else:
            raise NotImplementedError

    def _probability(self, probability_type, cardinality, importance, keep_idxs=None):
        if probability_type == "uniform":
            return np.ones(cardinality).astype(dtype="float64") / cardinality
        elif probability_type == "importance":
            imp = importance.numpy().astype(dtype="float64")
            return imp / np.sum(imp)
            # return np.exp(imp) / sum(np.exp(imp)) #Softmax
        elif probability_type == "radical":
            result = np.ones(cardinality).astype(dtype="float64")
            for indice in keep_idxs:
                result[indice] = cardinality / len(keep_idxs)
            return result / np.sum(result)
        else:
            raise NotImplementedError

    def _cost(self, weights0, weights1):
        if self.gpu_id != -1:
            weights0 = weights0.cuda(self.gpu_id)
            weights1 = weights1.cuda(self.gpu_id)

        norm0 = torch.norm(weights0, dim=-1, keepdim=True)
        norm1 = torch.norm(weights1, dim=-1, keepdim=True)
        if self.gpu_id != -1:
            norm0 = norm0.cuda(self.gpu_id)
            norm1 = norm1.cuda(self.gpu_id)

        distance = torch.cdist(
            weights0 / norm0, weights1 / norm1, p=self.p).cpu()

        weights0 = weights0.cpu()
        weights1 = weights1.cpu()
        return distance

    @torch.no_grad()
    def __call__(self, group, importance: torch.Tensor, pruning_idxs: torch.Tensor):
        """
        Calculates the Optimal Transport map.

        Args:
            group:  Group of dependent layers that have to be pruned in unison.
            importance: Importance score for each neural pairing.
            pruning_idxs: Indices of the neural pairings with the lowest importance score. E.g. if one wants to prune 16 neural pairings, len(pruning_idxs) = 16

        Returns:
            torch.Tensor: The Optimal Transport map
        """
        keep_idxs = None
        w_all = []

        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # out_channels
            if prune_fn == "prune_linear_out_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    if layer.bias:
                        torch.cat((w, layer.bias), dim=1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                w_all.append(w)

            # in_channels
            elif prune_fn == "prune_linear_in_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1)[idxs].flatten(1)
                w_all.append(w)

            if keep_idxs == None:
                keep_idxs = list(
                    set([i for i in range(w.shape[0])])
                    - set(int(i) for i in pruning_idxs)
                )

        print("w_all", len(w_all))
        if len(w_all) == 0:
            return

        w_all = torch.cat(w_all, dim=1)

        cost = None
        if self.target == "most_important":
            cost = self._cost(w_all, w_all[keep_idxs])
        else:
            gm = GaussianMixture(n_components=len(
                keep_idxs), random_state=0, covariance_type="spherical").fit(w_all)
            cost = self._cost(w_all, torch.from_numpy(gm.means_).float())

        source_prob = self._probability(
            self.source_probability, cost.shape[0], importance, keep_idxs
        )
        target_prob = self._probability(
            self.target_probability, cost.shape[1], importance[keep_idxs], keep_idxs
        )

        ot_map = ot.emd(
            source_prob, target_prob, cost.detach().cpu().numpy()
        ).transpose()

        ot_map = torch.from_numpy(ot_map).float()

        ot_map = ot_map / ot_map.sum(dim=0)
        print("w_all", w_all.shape)
        print("cost", cost.shape)
        print("source_prob", source_prob.shape)
        print("target_prob", target_prob.shape)

        return ot_map.float()


    def __str__(self):
        return f"OT_Source_{self.source_probability}_Target_{self.target_probability}"



def create_weight_aware_probability(importance_scores, channel_stats, keep_idxs, probability_type):
    """重み単位スコアを考慮した確率分布生成"""
    
    if probability_type == "uniform":
        return np.ones(len(importance_scores)).astype(dtype="float64") / len(importance_scores)
    
    elif probability_type == "importance":
        imp = importance_scores.numpy().astype(dtype="float64")
        return imp / np.sum(imp)
    
    elif probability_type == "weight_aware":
        # 複数の統計量を組み合わせた確率分布
        fo_component = channel_stats['fo_max'] + channel_stats['fo_std'] * 0.5
        snip_component = channel_stats['snip_max'] + channel_stats['snip_std'] * 0.5
        diversity_component = channel_stats['fo_important_ratio'] + channel_stats['snip_important_ratio']
        
        # 標準化
        fo_norm = (fo_component - fo_component.mean()) / (fo_component.std() + 1e-8)
        snip_norm = (snip_component - snip_component.mean()) / (snip_component.std() + 1e-8)
        div_norm = (diversity_component - diversity_component.mean()) / (diversity_component.std() + 1e-8)
        
        # 重み付き結合
        combined_score = fo_norm + snip_norm + div_norm * 0.3
        
        # 確率分布に変換（softmax）
        # print("combined_score", combined_score.shape)
        prob = torch.softmax(combined_score, dim=0).numpy().astype(dtype="float64")
        return prob
    
    elif probability_type == "max_preserving":
        # 最大値重視の確率分布（重要な重みを強く保護）
        max_scores = channel_stats['fo_max'] + channel_stats['snip_max']
        # print("max_scores", max_scores.shape)
        # keep_idxsが指定されている場合は、該当するインデックスのみ抽出
        if keep_idxs is not None:
            max_scores_subset = max_scores[keep_idxs]
            print(f"Debug max_preserving: max_scores_subset.shape={max_scores_subset.shape}, keep_idxs len={len(keep_idxs)}")
        else:
            max_scores_subset = max_scores
            
        prob = torch.softmax(max_scores_subset, dim=0).numpy().astype(dtype="float64")
        return prob
    
    elif probability_type == "diversity_aware":
        # 多様性考慮の確率分布
        diversity_scores = (channel_stats['fo_std'] + channel_stats['snip_std'] + 
                          channel_stats['fo_important_ratio'] + channel_stats['snip_important_ratio'])
        # keep_idxsが指定されている場合は、該当するインデックスのみ抽出
        if keep_idxs is not None:
            diversity_scores_subset = diversity_scores[keep_idxs]
        else:
            diversity_scores_subset = diversity_scores
            
        prob = torch.softmax(diversity_scores_subset, dim=0).numpy().astype(dtype="float64")
        return prob

def create_adaptive_importance_score(channel_stats, strategy="comprehensive"):
    """適応的重要度スコア生成"""
    
    if strategy == "conservative":
        # 保守的：最大値と一貫性を重視
        fo_score = channel_stats['fo_max'] * 0.4 + channel_stats['fo_consistency'] * 0.3 + channel_stats['fo_p95'] * 0.3
        snip_score = channel_stats['snip_max'] * 0.4 + channel_stats['snip_consistency'] * 0.3 + channel_stats['snip_p95'] * 0.3
        
    elif strategy == "aggressive":
        # 積極的：平均と構造的重要度を重視
        fo_score = channel_stats['fo_mean'] * 0.3 + channel_stats['fo_structural'] * 0.4 + channel_stats['fo_mutual_info'] * 0.3
        snip_score = channel_stats['snip_mean'] * 0.3 + channel_stats['snip_structural'] * 0.4 + channel_stats['snip_mutual_info'] * 0.3
        
    elif strategy == "comprehensive":
        # 包括的：全ての指標を使用
        fo_score = (channel_stats['fo_max'] * 0.2 + 
                   channel_stats['fo_mean'] * 0.15 +
                   channel_stats['fo_p95'] * 0.2 +
                   channel_stats['fo_consistency'] * 0.15 +
                   channel_stats['fo_structural'] * 0.15 +
                   channel_stats['fo_mutual_info'] * 0.15)
        
        snip_score = (channel_stats['snip_max'] * 0.2 +
                     channel_stats['snip_mean'] * 0.15 +
                     channel_stats['snip_p95'] * 0.2 +
                     channel_stats['snip_consistency'] * 0.15 +
                     channel_stats['snip_structural'] * 0.15 +
                     channel_stats['snip_mutual_info'] * 0.15)
    
    else:  # "balanced"
        # バランス：主要指標のみ
        fo_score = channel_stats['fo_max'] * 0.4 + channel_stats['fo_mean'] * 0.3 + channel_stats['fo_structural'] * 0.3
        snip_score = channel_stats['snip_max'] * 0.4 + channel_stats['snip_mean'] * 0.3 + channel_stats['snip_structural'] * 0.3
    
    # 標準化
    fo_norm = (fo_score - fo_score.mean()) / (fo_score.std() + 1e-8)
    snip_norm = (snip_score - snip_score.mean()) / (snip_score.std() + 1e-8)
    
    return fo_norm + snip_norm



class WeightAwareOptimalTransport:
    """重み単位スコアを考慮した最適輸送クラス"""
    
    def __init__(self, original_ot):
        self.original_ot = original_ot
        self.analyzer = SimpleRescueAnalyzer()
        
    def _weight_aware_probability(self, probability_type, cardinality, importance, 
                                weight_stats=None, keep_idxs=None):
        """重み単位統計を考慮した確率分布生成"""
        
        if weight_stats is None:
            # フォールバック：元の手法
            return self.original_ot._probability(probability_type, cardinality, importance, keep_idxs)
        
        if probability_type == "uniform":
            print("Using uniform probability distribution")
            return np.ones(cardinality).astype(dtype="float64") / cardinality
        
        elif probability_type == "weight_aware":
            print("Using weight-aware probability distribution")
            return create_weight_aware_probability(importance, weight_stats, keep_idxs, "weight_aware")
        
        elif probability_type == "max_preserving":
            print("Using max-preserving probability distribution")
            return create_weight_aware_probability(importance, weight_stats, keep_idxs, "max_preserving")
        
        elif probability_type == "diversity_aware":
            print("Using diversity-aware probability distribution")
            return create_weight_aware_probability(importance, weight_stats, keep_idxs, "diversity_aware")
        
        else:
            return self.original_ot._probability(probability_type, cardinality, importance, keep_idxs)
    
    @torch.no_grad()
    def __call__(self, group, importance: torch.Tensor, pruning_idxs: torch.Tensor, 
                 weight_stats: Dict = None, layer_idx: int = None):
        """重み単位統計を考慮した最適輸送マップ計算"""
        
        keep_idxs = None
        w_all = []

        # 既存の重み抽出処理（変更なし）
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn == "prune_linear_out_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    if layer.bias:
                        torch.cat((w, layer.bias), dim=1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                w_all.append(w)

            elif prune_fn == "prune_linear_in_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1)[idxs].flatten(1)
                w_all.append(w)

            if keep_idxs == None:
                keep_idxs = list(
                    set([i for i in range(w.shape[0])])
                    - set(int(i) for i in pruning_idxs)
                )

        if len(w_all) == 0:
            return

        w_all = torch.cat(w_all, dim=1)

        # コスト計算（既存）
        cost = self.original_ot._cost(w_all, w_all[keep_idxs])

        # 重み単位統計を考慮した確率分布生成
        # print("keep_idxs", keep_idxs)
        # print("importance", len(importance))
        source_prob = self._weight_aware_probability(
            self.original_ot.source_probability , cost.shape[0], importance, weight_stats
        )
        target_prob = self._weight_aware_probability(
            self.original_ot.target_probability, cost.shape[1], importance[keep_idxs], weight_stats, keep_idxs
        )

        # 最適輸送マップ計算
        # print("source_prob", source_prob.shape)
        # print("target_prob", target_prob.shape)
        # print("cost", cost.shape)
        # print("w_all", w_all.shape)
        # print("w_all", len(w_all))
        ot_map = ot.emd(
            source_prob, target_prob, cost.detach().cpu().numpy(), numThreads=32#, numItermax=1000000
        ).transpose()

        ot_map = torch.from_numpy(ot_map).float()
        ot_map = ot_map / ot_map.sum(dim=0)

        self.analyzer.analyze_layer_rescue(ot_map, pruning_idxs, keep_idxs, layer_idx)

        return ot_map.float()
    def get_rescue_summary(self):
        """救済効果サマリーを取得"""
        return self.analyzer.print_summary_report()
    

class EnhancedWeightAwareOptimalTransport:
    """改良版重み対応最適輸送"""
    
    def __init__(self, original_ot, importance_strategy="comprehensive", 
                 source_prob_type="entropy_regularized", target_prob_type="adaptive"):
        self.original_ot = original_ot
        self.importance_strategy = importance_strategy
        self.source_prob_type = source_prob_type
        self.target_prob_type = target_prob_type
    
    @torch.no_grad()
    def __call__(self, group, base_importance, pruning_idxs, channel_stats):
        """改良された最適輸送マップ計算"""
        
        # 既存の重み抽出処理（同じ）
        keep_idxs = None
        w_all = []
        
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn == "prune_linear_out_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                w_all.append(w)

            elif prune_fn == "prune_linear_in_channels":
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1)[idxs].flatten(1)
                w_all.append(w)

            if keep_idxs == None:
                keep_idxs = list(
                    set([i for i in range(w.shape[0])])
                    - set(int(i) for i in pruning_idxs)
                )

        if len(w_all) == 0:
            return

        w_all = torch.cat(w_all, dim=1)
        
        # 改良された重要度スコア
        enhanced_importance = create_adaptive_importance_score(
            channel_stats, strategy=self.importance_strategy
        )
        
        # コスト計算
        cost = self.original_ot._cost(w_all, w_all[keep_idxs])
        
        # 改良された確率分布生成
        source_prob = create_smart_probability_distribution(
            enhanced_importance, channel_stats, None, 
            prob_type=self.source_prob_type, temperature=0.5
        )
        
        target_prob = create_smart_probability_distribution(
            enhanced_importance, channel_stats, keep_idxs,
            prob_type=self.target_prob_type, temperature=1.0
        )
        
        # 最適輸送計算
        import ot
        ot_map = ot.emd(source_prob, target_prob, cost.detach().cpu().numpy(), numThreads=32).transpose()
        ot_map = torch.from_numpy(ot_map).float()
        ot_map = ot_map / ot_map.sum(dim=0)
        
        return ot_map.float()
    
def create_smart_probability_distribution(importance_scores, channel_stats, keep_idxs, 
                                        prob_type="adaptive", temperature=1.0):
    """スマートな確率分布生成"""
    
    if prob_type == "adaptive":
        # 重要度に応じて温度を調整
        if keep_idxs is not None:
            subset_scores = importance_scores[keep_idxs]
            # 重要度の分散が大きい場合は温度を下げる（よりシャープに）
            score_std = subset_scores.std()
            adaptive_temp = temperature * (1.0 + score_std)
            prob = torch.softmax(subset_scores / adaptive_temp, dim=0)
        else:
            score_std = importance_scores.std()
            adaptive_temp = temperature * (1.0 + score_std)
            prob = torch.softmax(importance_scores / adaptive_temp, dim=0)
            
    elif prob_type == "robust":
        # ロバストな確率分布（外れ値に対して頑健）
        if keep_idxs is not None:
            subset_scores = importance_scores[keep_idxs]
            # ランク変換でロバスト性を向上
            ranks = torch.argsort(torch.argsort(subset_scores)).float()
            prob = torch.softmax(ranks / temperature, dim=0)
        else:
            ranks = torch.argsort(torch.argsort(importance_scores)).float()
            prob = torch.softmax(ranks / temperature, dim=0)
            
    elif prob_type == "entropy_regularized":
        # エントロピー正則化
        if keep_idxs is not None:
            subset_scores = importance_scores[keep_idxs]
        else:
            subset_scores = importance_scores
            
        # 初期確率分布
        initial_prob = torch.softmax(subset_scores / temperature, dim=0)
        # エントロピー項を追加（多様性を促進）
        entropy_term = -initial_prob * torch.log(initial_prob + 1e-8)
        regularized_scores = subset_scores + 0.1 * entropy_term
        prob = torch.softmax(regularized_scores / temperature, dim=0)
        
    else:  # "standard"
        if keep_idxs is not None:
            subset_scores = importance_scores[keep_idxs]
        else:
            subset_scores = importance_scores
        prob = torch.softmax(subset_scores / temperature, dim=0)
    
    return prob.numpy().astype(dtype="float64")

class SimpleRescueAnalyzer:
    """シンプルな救済効果分析"""
    
    def __init__(self):
        self.layer_results = []
        self.total_rescued_ratio = 0.0
        self.total_dilution_ratio = 0.0
        self.total_layers = 0
    
    def analyze_layer_rescue(self, ot_map, pruning_idxs, keep_idxs, layer_idx):
        """
        単一層の救済効果分析
        
        Args:
            ot_map: 最適輸送マップ [keep_channels, total_channels]
            pruning_idxs: 削除対象のインデックス
            keep_idxs: 保持対象のインデックス
            layer_idx: 層番号
        
        Returns:
            dict: 救済分析結果
        """
        
        total_channels = ot_map.shape[1]
        keep_channels = len(keep_idxs)
        prune_channels = len(pruning_idxs)
        
        # 1. 削除対象の救済率
        # 削除対象→保持対象への輸送量の合計
        pruned_to_kept = ot_map[:, pruning_idxs]  # [keep_channels, prune_channels]
        rescued_weights = pruned_to_kept.sum().item()  # 削除対象から救済された総量
        max_possible_rescue = prune_channels  # 最大救済可能量（全て救済された場合）
        
        rescue_ratio = rescued_weights / max_possible_rescue if max_possible_rescue > 0 else 0.0
        
        # 2. 保持対象の希釈率  
        # 保持対象→保持対象への輸送量（本来の寄与）
        kept_to_kept = ot_map[:, keep_idxs]  # [keep_channels, keep_channels]
        preserved_weights = torch.diagonal(kept_to_kept).sum().item()  # 対角成分の合計
        max_possible_preserve = keep_channels  # 最大保持可能量（希釈なしの場合）
        
        dilution_ratio = 1.0 - (preserved_weights / max_possible_preserve) if max_possible_preserve > 0 else 0.0
        
        result = {
            'layer_idx': layer_idx,
            'total_channels': total_channels,
            'pruned_channels': prune_channels,
            'kept_channels': keep_channels,
            'rescued_ratio': rescue_ratio,  # 削除対象の何割が救済されたか
            'dilution_ratio': dilution_ratio,  # 保持対象の何割が希釈されたか
            'rescued_absolute': rescued_weights,  # 救済された絶対量
            'preserved_absolute': preserved_weights  # 保持された絶対量
        }
        
        self.layer_results.append(result)
        
        print(f"Layer {layer_idx:2d}: Rescued {rescue_ratio:.3f} ({rescue_ratio*100:.1f}%) of pruned channels, "
              f"Diluted {dilution_ratio:.3f} ({dilution_ratio*100:.1f}%) of kept channels")
        
        return result
    
    def compute_global_rescue_effect(self):
        """
        モデル全体での救済効果を計算
        
        Returns:
            dict: グローバル救済分析結果
        """
        
        if not self.layer_results:
            print("No layer results available for global analysis.")
            return {}
        
        # 各層の結果を重み付き平均で集約（チャネル数で重み付け）
        total_pruned = sum(r['pruned_channels'] for r in self.layer_results)
        total_kept = sum(r['kept_channels'] for r in self.layer_results)
        total_rescued_absolute = sum(r['rescued_absolute'] for r in self.layer_results)
        total_preserved_absolute = sum(r['preserved_absolute'] for r in self.layer_results)
        
        # グローバル救済率：全削除対象のうち何割が救済されたか
        global_rescue_ratio = total_rescued_absolute / total_pruned if total_pruned > 0 else 0.0
        
        # グローバル希釈率：全保持対象のうち何割が希釈されたか  
        global_dilution_ratio = 1.0 - (total_preserved_absolute / total_kept) if total_kept > 0 else 0.0
        
        # 層ごとの救済率の分布統計
        rescue_ratios = [r['rescued_ratio'] for r in self.layer_results]
        dilution_ratios = [r['dilution_ratio'] for r in self.layer_results]
        
        global_result = {
            'total_layers': len(self.layer_results),
            'total_pruned_channels': total_pruned,
            'total_kept_channels': total_kept,
            'global_rescue_ratio': global_rescue_ratio,
            'global_dilution_ratio': global_dilution_ratio,
            'rescue_ratio_mean': np.mean(rescue_ratios),
            'rescue_ratio_std': np.std(rescue_ratios),
            'rescue_ratio_min': np.min(rescue_ratios),
            'rescue_ratio_max': np.max(rescue_ratios),
            'dilution_ratio_mean': np.mean(dilution_ratios),
            'dilution_ratio_std': np.std(dilution_ratios),
            'dilution_ratio_min': np.min(dilution_ratios),
            'dilution_ratio_max': np.max(dilution_ratios)
        }
        
        return global_result
    
    def print_summary_report(self):
        """サマリーレポートを出力"""
        
        global_result = self.compute_global_rescue_effect()
        
        if not global_result:
            return
        
        print("\n" + "="*70)
        print("RESCUE EFFECT SUMMARY REPORT")
        print("="*70)
        
        print(f"\nModel Overview:")
        print(f"  Total layers analyzed: {global_result['total_layers']}")
        print(f"  Total pruned channels: {global_result['total_pruned_channels']:,}")
        print(f"  Total kept channels: {global_result['total_kept_channels']:,}")
        
        print(f"\nGlobal Rescue Effect:")
        print(f"  Rescued ratio: {global_result['global_rescue_ratio']:.4f} ({global_result['global_rescue_ratio']*100:.2f}%)")
        print(f"  → {global_result['global_rescue_ratio']*100:.2f}% of pruned channels contribute to the compressed model")
        
        print(f"\nGlobal Dilution Effect:")
        print(f"  Dilution ratio: {global_result['global_dilution_ratio']:.4f} ({global_result['global_dilution_ratio']*100:.2f}%)")
        print(f"  → {global_result['global_dilution_ratio']*100:.2f}% of kept channels' original contribution is sacrificed for rescue")
        
        print(f"\nPer-Layer Statistics:")
        print(f"  Rescue ratio - Mean: {global_result['rescue_ratio_mean']:.4f}, Std: {global_result['rescue_ratio_std']:.4f}")
        print(f"  Rescue ratio - Range: [{global_result['rescue_ratio_min']:.4f}, {global_result['rescue_ratio_max']:.4f}]")
        print(f"  Dilution ratio - Mean: {global_result['dilution_ratio_mean']:.4f}, Std: {global_result['dilution_ratio_std']:.4f}")
        print(f"  Dilution ratio - Range: [{global_result['dilution_ratio_min']:.4f}, {global_result['dilution_ratio_max']:.4f}]")
        
        # 効果の解釈
        print(f"\nInterpretation:")
        rescue_pct = global_result['global_rescue_ratio'] * 100
        dilution_pct = global_result['global_dilution_ratio'] * 100
        
        if rescue_pct > 50:
            print(f"  ✓ Good rescue effect: {rescue_pct:.1f}% of pruned information is preserved")
        elif rescue_pct > 20:
            print(f"  ~ Moderate rescue effect: {rescue_pct:.1f}% of pruned information is preserved") 
        else:
            print(f"  ✗ Limited rescue effect: Only {rescue_pct:.1f}% of pruned information is preserved")
        
        if dilution_pct < 20:
            print(f"  ✓ Low dilution cost: Only {dilution_pct:.1f}% of kept channels are diluted")
        elif dilution_pct < 50:
            print(f"  ~ Moderate dilution cost: {dilution_pct:.1f}% of kept channels are diluted")
        else:
            print(f"  ✗ High dilution cost: {dilution_pct:.1f}% of kept channels are diluted")
        
        print("="*70)
        
        return global_result
    