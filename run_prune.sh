#!/bin/bash
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
m_list=(128)  
n=0 
p=8
q=1
dataset="ptb"
# "c4"
# "openwebtext"
# 
# "ptb"
# "rte"
# "boolq"
# "wikitext2"
cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device
export WANDB_MODE="dryrun"
prune_method="cim"
# "wanda"

groupings=("column-wise")
# ou-base  

for m in "${m_list[@]}"; do
    n=$m  # Set n equal to m

    echo "Starting pruning with m=$m, n=$n, p=$p, q=$q"

    for grouping in "${groupings[@]}"; do
        echo "Pruning method: $prune_method, Grouping: $grouping, Sparsity Ratio: $sparsity_ratio, m: $m, n: $n, p: $p, q: $q"

        output_dir="out/new_alpha0/model_results/${dataset}/${prune_method}/${sparsity_ratio}/${grouping}_${m}_${n}_test"
        mkdir -p "$output_dir"

        run_python_command() {
            python /home/snakkill/llm_int/crxb/farshad/wanda/main_rte.py \
                --model "$model" \
                --prune_method "$prune_method" \
                --grouping "$grouping" \
                --sparsity_ratio "$sparsity_ratio" \
                --sparsity_type "unstructured" \
                --save "$output_dir" \
                --save_model "${output_dir}/pruned_model" \
                --m "$m" \
                --n "$n" \
                --p "$p" \
                --q "$q" \
                --dataset "$dataset" \
                --nsamples 64
        }

        echo "Running pruning: $prune_method with grouping: $grouping and sparsity ratio: $sparsity_ratio"
        run_python_command
        echo "Finished pruning: $prune_method with grouping: $grouping and sparsity ratio: $sparsity_ratio"
    done

    echo "Completed all groupings for m=$m, n=$n"
    echo "============================================="
done

echo "All pruning experiments completed successfully."

exit
