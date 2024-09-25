# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.1 --seed 0 --compute_pretrained_outputs --precision 16 --devices 0 1 --strategy fsdp

python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.1 --seed 0 --devices 0 1 2 --strategy fsdp --compute_pretrained_outputs


# for seed in 0 1 2 3 4
# do
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.1 --seed $seed --devices 0 1 --strategy fsdp
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.08 --seed $seed --devices 0 1 --strategy fsdp
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.06 --seed $seed --devices 0 1 --strategy fsdp
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.04 --seed $seed --devices 0 1 --strategy fsdp
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.02 --seed $seed --devices 0 1 --strategy fsdp
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.01 --seed $seed --devices 0 1 --strategy fsdp
# done

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.1 --seed 0 --compute_pretrained_outputs --precision 16 --strategy fsdp

# for seed in 0 1 2 3 4
# do
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.1 --seed $seed
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.08 --seed $seed
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.06 --seed $seed
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.04 --seed $seed
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.02 --seed $seed
# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 200\
#     --scale 0.01 --seed $seed
# done
