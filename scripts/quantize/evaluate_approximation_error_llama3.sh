

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --compute_pretrained_outputs --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 1000 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --use_qlora

for seed in 0 1 2 
do

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --scale 0.1 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --scale 0.08 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --scale 0.06 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --scale 0.04 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "../llama/llama-3/Meta-Llama-3-8B-hf" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=2\
    --scale 0.02 --seed $seed --use_qlora
done