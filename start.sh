# Set common variables
model="meta-llama/Meta-Llama-3-8B"
# model="lmsys/vicuna-13b-v1.5"

python main.py \
--model $model \
--prune_method "structured_afr_with_intrafusion" \
--pruning_ratio 0.5 \
--strategy_idx 0 \
--nsamples 128 \
--save_model "./pruned_model/trash" \
# --pruning_ration : 枝刈り率
