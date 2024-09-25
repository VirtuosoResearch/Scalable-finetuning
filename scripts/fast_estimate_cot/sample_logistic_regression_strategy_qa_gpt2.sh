python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key gpt2 --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_meta_train/epoch_epoch=4 --batch_size 8\
    --run 0 --project_dim 100 --device 2 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --subset_size 0.5 --scale 0.05\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_project_100_subset_size_0.5_clusters_100
