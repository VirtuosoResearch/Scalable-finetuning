python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_new_run_0/epoch_epoch=17\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 100 --device 2 \
    --load_clusters --num_clusters 100 --number_of_subsets 1000 --subset_size 0.25 --scale 0.4
