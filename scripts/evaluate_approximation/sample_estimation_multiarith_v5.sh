python fast_estimate_linear_regression.py --dataset_key multiarith --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_multiarith_ft_cot_lora_r_4_run_4/epoch_epoch=19\
    --train_lora --lora_rank 4 --lora_alpha 32 --run 4 --device 1\
    --number_of_subsets 2500 --subset_size 0.75 \
    --load_sample_task_dir multiarith_flan_t5_base_ft_cot_run_0