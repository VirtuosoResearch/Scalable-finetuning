python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.1 --seed 0 --compute_pretrained_outputs --precision 16 --devices 2 --strategy auto

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.2 --seed $seed --devices 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.16 --seed $seed --devices 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.12 --seed $seed --devices 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.08 --seed $seed --devices 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.04 --seed $seed --devices 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 1000\
    --scale 0.02 --seed $seed --devices 2 --strategy auto
done


# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.1 --seed 0 --devices 2 --strategy auto