python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 1000 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0

for seed in 0 1 2 
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.2 --seed $seed

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.16 --seed $seed

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.12 --seed $seed

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.08 --seed $seed

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.04 --seed $seed

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.02 --seed $seed
done

for seed in 0 1 2 
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.2 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.16 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.12 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.08 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.04 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name Alpaca_google-gemma-fast_approximation_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_google-gemma-2b_lora_r_32_meta_initialization_alpaca_run_0_epoch_0\
    --scale 0.02 --seed $seed --use_qlora
done