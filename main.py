import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from lib.prune import check_sparsity, Structured_AFR_with_IntraFusion, enhanced_structured_afr_with_intrafusion, enhanced_structured_afr_v2
from lib.eval import eval_ppl

print('torch', version('torch'))  # 2.1.0
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument("--prune_method", type=str, default="cfsp", choices=["structured_afr_with_intrafusion","done"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--strategy_idx',type=int, default=0, choices=[0,1,2])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if torch.cuda.is_available():
        print(" ---- CUDA is available! ------")
    else:
        print(" ---- no cuda! ------")

    # Prune the model
    print("pruning starts")
    if args.prune_method == "structured_afr_with_intrafusion":
        print(f"loading llm model {args.model}, pruning method: {args.prune_method}")
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.seqlen = 1024
        device = torch.device("cuda:0")
        # device = torch.cuda.device_count()
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        enhanced_structured_afr_with_intrafusion(args, model, tokenizer, device)
        # Structured_AFR_with_IntraFusion(args, model, tokenizer, device)
        # enhanced_structured_afr_v2(args, model, tokenizer, device)
    elif args.prune_method == "done":
        print(f"loading llm model {args.model} with pruned model")
        model_path = "./pruned_model/Llama3-8B_AFR-St_0.5p_onlyFFN_sizeChanged"
        print("evaluate :", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
        model.seqlen = 1024
        device = torch.device("cuda:0")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    pruned_model_param = check_sparsity(model)
    print(f"model parameter {pruned_model_param}B")
    print("*"*30)
    
    if args.eval:
        print("Start evaluation")
        ppl = eval_ppl(args, model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")

    if args.save_model and args.prune_method != "done":
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
