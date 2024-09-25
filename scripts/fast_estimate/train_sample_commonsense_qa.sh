python sample_train_results.py --dataset_key commonsense_qa --model_key flan_t5_base --preset_key ft_cot\
    --load_model_dir flan_t5_base_commonsense_qa_ft_cot_lora_r_16_new_run_0/epoch_epoch=8\
    --train_lora --lora_rank 16 --lora_alpha 128 --epochs 5\
    --project_dim 200 --device 1 \
    --load_sample_task_dir commonsense_qa_flan_t5_base_ft_cot_run_0_scale_0.4_clusters_100\
    --load_clusters --num_clusters 100 --scale 0.4 --save_name true_performance