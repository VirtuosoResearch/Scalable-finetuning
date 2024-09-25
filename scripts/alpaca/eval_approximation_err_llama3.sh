python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name test

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.2 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.16 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.12 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 0 1 2 --strategy auto
done
