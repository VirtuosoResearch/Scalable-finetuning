python fast_estimate_cluster_gradients.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --run 0 --project_dim 200 --device 0 \
    --num_clusters 100

python fast_estimate_cluster_gradients.py --dataset_key commonsense_qa --model_key flan_t5_base --preset_key ft_cot\
    --run 0 --project_dim 200 --device 0 \
    --num_clusters 100