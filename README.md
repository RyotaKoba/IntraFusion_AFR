# 最適輸送によるモデル軽量化の実験

LLM（大規模言語モデル）に対して、最適輸送理論を用いた構造化プルーニング手法を適用するための実験コードです。

## 概要

このプロジェクトは、Structured AFR（Activation-based Feature Reweighting）とIntra-Fusionを組み合わせた手法により、LLMの中間層（FFN層）を効率的に圧縮します。最適輸送マップを用いて、削除対象のチャネル情報を保持対象のチャネルに融合することで、性能劣化を抑えながらモデルサイズを削減します。

## 主な特徴

- **最適輸送ベースの重み融合**: 削除対象の重要な情報を保持チャネルに再配分
- **複合的重要度評価**: Fisher Observation（FO）とSNIPスコアの組み合わせ
- **重み単位統計の活用**: チャネル単位だけでなく、個別の重み要素も考慮した高度な分析
- **救済効果の可視化**: 削除対象がどの程度救済されたかを定量的に分析

## 環境構築

### 方法1: Singularityコンテナを使用（推奨）

Singularityコンテナイメージ（.sifファイル）をGoogle Driveから取得できます。

**ダウンロードリンク**: `[YOUR_GOOGLE_DRIVE_URL_HERE]`
```bash
# Google Driveからダウンロード後、zipを解凍
unzip rkoba_pruning.zip

# Singularityコンテナで実行
singularity exec --nv rkoba_pruning.sif sh start.sh
```

**注意**: 
- `.sif`ファイルは`.gitignore`に含まれており、リポジトリには含まれていません
- コンテナイメージには必要な全ての依存パッケージがプリインストールされています
- `--nv`オプションでNVIDIA GPUサポートを有効化します

### 方法2: ローカル環境でのセットアップ
```bash
# 必要なパッケージのインストール
pip install torch transformers datasets accelerate
pip install POT  # Python Optimal Transport
pip install scikit-learn numpy tqdm
```

### 推奨環境

- Python 3.8以上
- PyTorch 2.0以上
- CUDA対応GPU（推奨：メモリ24GB以上）
- Singularity 3.5以上（コンテナ使用時）

## 使用方法

### Singularityコンテナでの実行
```bash
# Slurmクラスタでの実行
sbatch slurm.sh

# 直接実行
singularity exec --nv rkoba_pruning.sif python main.py \
  --model meta-llama/Meta-Llama-3-8B \
  --prune_method structured_afr_with_intrafusion \
  --pruning_ratio 0.5 \
  --nsamples 128 \
  --save_model ./pruned_model/llama3-8b-pruned
```

### ローカル環境での実行
```bash
python main.py \
  --model meta-llama/Meta-Llama-3-8B \
  --prune_method structured_afr_with_intrafusion \
  --pruning_ratio 0.5 \
  --nsamples 128 \
  --save_model ./pruned_model/llama3-8b-pruned
```

### 主要な引数

- `--model`: HuggingFaceのモデル名またはパス
- `--prune_method`: プルーニング手法
  - `structured_afr_with_intrafusion`: 提案手法
  - `done`: プルーニング済みモデルの評価のみ
- `--pruning_ratio`: プルーニング率（0.0〜1.0）
- `--nsamples`: キャリブレーションデータのサンプル数
- `--save_model`: プルーニング後のモデル保存先
- `--eval`: プルーニング後にperplexityを評価
- `--strategy_idx`: 最適輸送の戦略選択（0:包括的, 1:保守的, 2:バランス型）
- `--cache_dir`: モデルキャッシュディレクトリ（デフォルト: `llm_weights`）

### プルーニング済みモデルの評価
```bash
python main.py \
  --prune_method done \
  --eval
```

## プロジェクト構造
```
.
├── main.py                    # メインスクリプト
├── start.sh                   # 実行用シェルスクリプト
├── slurm.sh                   # Slurm用ジョブスクリプト
├── README.md                  # このファイル
├── .gitignore                 # Git除外設定
├── rkoba_pruning.sif          # Singularityイメージ（Google Driveから取得）
├── pruned_model/              # プルーニング済みモデル保存先（除外）
├── llm_weights/               # モデルキャッシュディレクトリ（除外）
└── lib/
    ├── data.py                # データローダー（WikiText-2）
    ├── eval.py                # モデル評価（perplexity計算）
    ├── model.py               # モデル関連ユーティリティ
    ├── prune.py               # プルーニングメイン処理
    ├── optimal_transport.py   # 最適輸送マップ計算
    ├── ops.py                 # カスタムオペレーション定義
    └── pruner/
        └── function.py        # 層ごとのプルーニング関数
```

## アルゴリズム詳細

### 1. 重要度スコアの計算

各チャネルの重要度を以下の2つの指標で評価します：

- **Fisher Observation (FO)**: 特徴マップの特異値に基づく重要度
- **SNIP**: 勾配と重みの積に基づく重要度

両指標を標準化して結合することで、ロバストな重要度評価を実現します。

### 2. 最適輸送マップの生成

削除対象チャネルと保持チャネル間の最適な輸送マップを計算します。以下の確率分布を選択可能：

- `uniform`: 一様分布
- `importance`: 重要度に基づく分布
- `weight_aware`: 重み単位統計を考慮した分布
- `max_preserving`: 最大値保護型分布
- `diversity_aware`: 多様性考慮型分布

### 3. Intra-Fusionによる重み融合

最適輸送マップを用いて、以下の3つのプロジェクション層の重みを融合：

- `gate_proj`: ゲート投影層（出力次元方向に融合）
- `up_proj`: アップ投影層（出力次元方向に融合）
- `down_proj`: ダウン投影層（入力次元方向に融合）

融合は以下の式で計算されます：
```
W_fused = OT_map @ W_original
```

### 4. 救済効果の分析

各層およびモデル全体で以下の指標を計算：

- **救済率（Rescue Ratio）**: 削除対象のうち情報が保存された割合
- **希釈率（Dilution Ratio）**: 保持対象が希釈された割合

## 実験結果の例
```
Layer 00: Rescued 0.234 (23.4%) of pruned channels, Diluted 0.156 (15.6%) of kept channels
Layer 01: Rescued 0.267 (26.7%) of pruned channels, Diluted 0.178 (17.8%) of kept channels
...

======================================================================
RESCUE EFFECT SUMMARY REPORT
======================================================================

Model Overview:
  Total layers analyzed: 32
  Total pruned channels: 65,536
  Total kept channels: 65,536

Global Rescue Effect:
  Rescued ratio: 0.2456 (24.56%)
  → 24.56% of pruned channels contribute to the compressed model

Global Dilution Effect:
  Dilution ratio: 0.1623 (16.23%)
  → 16.23% of kept channels' original contribution is sacrificed for rescue

Per-Layer Statistics:
  Rescue ratio - Mean: 0.2456, Std: 0.0234
  Rescue ratio - Range: [0.1890, 0.3012]
  Dilution ratio - Mean: 0.1623, Std: 0.0187
  Dilution ratio - Range: [0.1234, 0.2101]

Interpretation:
  ~ Moderate rescue effect: 24.6% of pruned information is preserved
  ✓ Low dilution cost: Only 16.2% of kept channels are diluted
======================================================================
```

## カスタマイズ

### 新しい確率分布の追加

`lib/optimal_transport.py`の`create_weight_aware_probability`関数に新しい分布タイプを追加できます：
```python
elif probability_type == "your_custom_type":
    # 独自の確率分布計算
    custom_scores = your_calculation(channel_stats)
    prob = torch.softmax(custom_scores, dim=0).numpy().astype(dtype="float64")
    return prob
```

### 重要度計算手法の変更

`lib/prune.py`の`calculate_weight_level_importance`関数で、独自の重要度指標を実装できます。

### 新しい戦略の追加

`lib/prune.py`の`enhanced_structured_afr_v2`関数で、戦略リストに追加：
```python
strategies = [
    ("comprehensive", "entropy_regularized", "adaptive"),
    ("conservative", "robust", "adaptive"), 
    ("balanced", "adaptive", "robust"),
    ("your_strategy", "your_source_prob", "your_target_prob")  # 追加
]
```

## 注意事項

- **メモリ要件**: モデルのロードには大量のメモリが必要です
  - Llama-3-8B: 約16GB
  - Llama-3-70B: 約140GB（マルチGPU推奨）
- **実行時間**: プルーニング処理は数時間かかる場合があります
- **プルーニング率**: `pruning_ratio`を0.5以上にすると、性能が大幅に劣化する可能性があります
- **データセット**: WikiText-2が自動でダウンロードされます（初回のみ）

## Singularityコンテナの詳細

### コンテナに含まれるパッケージ

- Python 3.10
- PyTorch 2.1.0（CUDA 11.8対応）
- Transformers（最新版）
- POT（Python Optimal Transport）
- その他必要な依存パッケージ

### コンテナの再ビルド

独自の環境が必要な場合は、Singularity定義ファイルから再ビルドできます：
```bash
# 定義ファイルの作成（例）
sudo singularity build rkoba_pruning.sif your_definition.def
```

## トラブルシューティング

### メモリ不足エラー
```bash
# 解決策1: サンプル数を減らす
--nsamples 64

# 解決策2: より小さいモデルで試す
--model meta-llama/Meta-Llama-3-1B

# 解決策3: float16を使用（要コード修正）
torch_dtype=torch.float16
```

### 最適輸送の収束エラー
```bash
# 解決策1: プルーニング率を調整
--pruning_ratio 0.3

# 解決策2: 確率分布タイプを変更
# lib/optimal_transport.pyで変更
target_probability="uniform"
source_probability="importance"
```

### Singularityコンテナの権限エラー
```bash
# ユーザー権限で実行
singularity exec --nv rkoba_pruning.sif [command]

# rootless modeで実行
singularity exec --nv --userns rkoba_pruning.sif [command]
```

### CUDA関連エラー
```bash
# GPUが認識されているか確認
singularity exec --nv rkoba_pruning.sif python -c "import torch; print(torch.cuda.is_available())"

# CUDAバージョンの確認
nvidia-smi
```

## 参考文献

- **SparseGPT**: https://github.com/IST-DASLab/sparsegpt
- **Python Optimal Transport (POT)**: https://pythonot.github.io/
- **Optimal Transport for structured pruning**: [論文リンク予定]

## 今後の改善予定

- [ ] マルチGPU対応の強化（DDP、DeepSpeedなど）
- [ ] より多様なモデルアーキテクチャへの対応（GPT、BERT系など）
- [ ] 自動ハイパーパラメータチューニング機能
- [ ] 詳細な性能ベンチマーク結果の追加
- [ ] 量子化との組み合わせ実験
- [ ] より高速な最適輸送アルゴリズムの導入

---

**重要**: Google Driveのリンクは各自で編集してください：
- Singularityイメージ（.sif）: `[YOUR_GOOGLE_DRIVE_URL_HERE]`
