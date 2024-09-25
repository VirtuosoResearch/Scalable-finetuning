python fast_estimate_linear_regression.py --dataset_key multiarith --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_multiarith_ft_cot_t70_64aug_lora_r_4_run_0/epoch_epoch=19\
    --train_lora --lora_rank 4 --lora_alpha 32 \
    --run 0 --project_dim 200 --device 1\
    --load_clusters --num_clusters 100 --number_of_subsets 1000 --scale 0.4