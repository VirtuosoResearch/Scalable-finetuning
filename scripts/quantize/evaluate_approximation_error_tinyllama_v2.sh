python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 16 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --compute_pretrained_outputs --save_name quantized_meta_init_gradients_ \
    --downsample 400 --num_batches_gradients 1000 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --use_qlora

for seed in 0 1 2 
do

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.1 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.08 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.06 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.04 --seed $seed --use_qlora

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.02 --seed $seed --use_qlora
done

python fast_estimate_eval_approximation_alpaca.py \
    --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 16 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --save_name quantized_meta_init_gradients \
    --downsample 400 --num_batches_gradients 200 --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_0\
    --scale 0.1 --seed 1 --use_qlora