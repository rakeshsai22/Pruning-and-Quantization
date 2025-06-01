import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# from huggingface_hub import interpreter_login
# interpreter_login()
import numpy as np
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_cim, prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
# from lib.prune import prune_cim_magnitude, prune_cim, prune_cim_hessian, prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
# from lib.prune_cim import 
# from lib.prune_cimh import prune_cim2
# from lib.eval import eval_ppl, eval_zero_shot
# from lib.evalboth import eval_ppl
from lib.eval_both import eval_ppl_wikitext, eval_ppl_ptb

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

access_token = "### access token ###"


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_auth_token=access_token
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["cim", "magnitude", "wanda", "sparsegpt",
                                                             "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter",
                                                             "ablate_wanda_iter", "search",
                                                             'cim_hessian', 'cim_magnitude'])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")

    ################### group shapes ###################
    parser.add_argument('--grouping', type=str, required=True, choices=['column-wise', 'row-wise', 'ou-base'], help="Grouping method for pruning")
    parser.add_argument('--m', type=int, required=True, help="Size parameter m")
    parser.add_argument('--n', type=int, required=True, help="Size parameter n")
    parser.add_argument('--p', type=int, required=False, help="Size parameter p for 'ou-base'", default=1)
    parser.add_argument('--q', type=int, required=False, help="Size parameter q for 'ou-base'", default=1)
    parser.add_argument('--dataset', type=str, required=True,choices=['wikitext2', 'c4', 'ptb','boolq','openwebtext'], help="dataset for perplexity")
    
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.30, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"model_name: {model_name}")
    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    print(f"model {model}")
    model.eval()
    #tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Load tokenizer from the same path as the model
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        use_auth_token=" ## access token ## "
    )

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "cim" in args.prune_method:
            print("calling prune_cim")
            prune_cim(args, model, tokenizer, device)
        elif args.prune_method == 'cim_hessian':
            print(f"args: {args}")
            print("calling prune_cim_hessian")
            # prune_cim_hessian(args, model, tokenizer, device)
        # elif args.prune_method == 'cim_hessian':
        #     prune_cim2(args, model, tokenizer, device,128,128,8,1)
        elif args.prune_method == 'cim_magnitude':
            print(f"args: {args}")
            print("calling prune_cim_magnitude")
            # prune_cim_magnitude(args, model, tokenizer, device)
        
    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    # ppl_test = eval_ppl_wikitext(args, model, tokenizer, device)
    #### ppl test ###################################################
    
    # ppl_test = eval_ppl_ptb(args, model, tokenizer, device)
    ppl_test = 0.000
    print(f"perplexity {ppl_test}")
    ################################################################



    ### eval edit for both datasets #############################################################
    # perplexities = eval_ppl(args, model, tokenizer, device)
    # for dataset_name, ppl in perplexities.items():
    #     print(f"{dataset_name} perplexity: {ppl}")

    
    ################################################################ 

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test\telapsed_time(sec)", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{elapsed_time}", file=f, flush=True)
    # with open(save_filepath, "w") as f:
    #     print("method\tactual_sparsity\tppl_test", file=f, flush=True)
    #     print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{perplexities:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        num_shot = 0
        # results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        # print(results)

    if args.save_model:
                # Convert only the pruned layers to sparse before saving
        # for name, param in model.named_parameters():
        #     if "weight" in name and param.requires_grad:  # Apply only to weight matrices
        #         param.data = param.data.to_sparse()

        # # Save the sparse model
        # torch.save(model.state_dict(), args.save_model)
        print("Save Model")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
