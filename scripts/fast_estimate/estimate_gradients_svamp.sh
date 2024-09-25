python fast_estimate_compute_gradients.py --dataset_key svamp --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_svamp_ft_cot_t70_64aug_lora_r_8_run_0/epoch_epoch=19\
    --train_lora --lora_rank 8 --lora_alpha 64 \
    --run 0 --project_dim 200 --device 0

# python fast_estimate_compute_gradients.py --dataset_key svamp --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
#     --load_model_dir flan_t5_base_svamp_ft_cot_t70_64aug_lora_r_8_run_1/epoch_epoch=19\
#     --train_lora --lora_rank 8 --lora_alpha 64 \
#     --run 1 --project_dim 200 --device 0