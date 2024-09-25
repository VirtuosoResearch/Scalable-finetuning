python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 0 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_t70_64aug_forward_selection\
    --save_name forward_selection --runs 2

python train_forward_selection.py --dataset_key commonsense_qa --model_key flan_t5_base --preset_key ft_cot\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --device 0 --save_name forward_selection\
    --load_clusters --num_clusters 100 --project_dim 200