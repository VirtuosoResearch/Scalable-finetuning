python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_new_run_1/epoch_epoch=13\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 2 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_0/epoch_epoch=13\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 3 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_1/epoch_epoch=8\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 4 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_2/epoch_epoch=7\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 5 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_3/epoch_epoch=15\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 6 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_4/epoch_epoch=7\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 7 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100

python fast_estimate_linear_regression.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_run_5/epoch_epoch=14\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 8 --project_dim 100 --device 1 \
    --load_clusters --num_clusters 100 --number_of_subsets 100 --scale 0.4\
    --load_sample_task_dir strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100