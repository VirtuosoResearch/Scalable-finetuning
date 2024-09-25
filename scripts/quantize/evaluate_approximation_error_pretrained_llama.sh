# "../llama/llama-3/Meta-Llama-3-8B-hf"

for model_key in "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 1 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --compute_pretrained_outputs --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 1000 \
    --use_qlora

for seed in 0 1 2 3 4
do

python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.1 --seed $seed --use_qlora --abs_scale 15

python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.08 --seed $seed --use_qlora --abs_scale 15

python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.06 --seed $seed --use_qlora --abs_scale 15

python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.04 --seed $seed --use_qlora --abs_scale 15

python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 --strategy auto --save_name quantized_pretrained_gradients \
    --downsample 400 --num_batches_gradients 200 \
    --scale 0.02 --seed $seed --use_qlora --abs_scale 15
done
done