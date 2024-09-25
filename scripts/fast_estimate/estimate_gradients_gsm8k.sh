python fast_estimate_compute_gradients.py --dataset_key gsm8k --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_gsm8k_ft_cot_lora_r_16_run_0/epoch_epoch=19\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 200 --device 1
    # --load_clusters --num_clusters 200 --number_of_subsets 1

python fast_estimate_compute_gradients.py --dataset_key gsm8k --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_gsm8k_ft_cot_lora_r_16_run_1/epoch_epoch=19\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 1 --project_dim 200 --device 1

python fast_estimate_compute_gradients.py --dataset_key gsm8k --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_gsm8k_ft_cot_lora_r_16_run_2/epoch_epoch=19\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 2 --project_dim 200 --device 1