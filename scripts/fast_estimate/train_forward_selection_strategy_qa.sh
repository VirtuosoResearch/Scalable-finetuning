python train_forward_selection.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --device 1 --save_name forward_selection\
    --load_clusters --num_clusters 100 --project_dim 200