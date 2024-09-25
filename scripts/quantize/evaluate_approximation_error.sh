for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.2 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.16 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.12 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.08 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.04 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4\
    --scale 0.02 --seed $seed --use_qlora
done