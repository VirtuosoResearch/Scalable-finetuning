# additional step: use export_hf_model.py 
# specify model_key, lora_rank, lora_alpha, load_model_dir, and save_name_dir

save_name_dir="Alpaca_google-gemma-2b_lora_r_32_quantized_meta_initialization_run_0_epoch_4"
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 32 --lora_alpha 128 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 2 --strategy auto --compute_pretrained_outputs --save_name fast_approximation_quantized_gradients \
    --downsample 400 --num_batches_gradients 1000 --load_model_dir $save_name_dir\
    --use_qlora