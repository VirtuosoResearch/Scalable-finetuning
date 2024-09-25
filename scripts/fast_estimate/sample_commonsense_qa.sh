python fast_estimate_linear_regression.py --dataset_key commonsense_qa --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_commonsense_qa_ft_cot_lora_r_16_new_run_0/epoch_epoch=8\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 200 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4