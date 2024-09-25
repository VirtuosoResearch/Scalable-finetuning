python sample_train_results.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --project_dim 100 --device 2 --epochs 5\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_project_100_subset_size_0.5_clusters_100\
    --load_clusters --num_clusters 100 --scale 0.4 --save_name sample_train