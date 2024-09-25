python custom_train.py --dataset_key strategy_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 2 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir strategy_qa_ft_cot_t70_64aug_forward_selection\
    --save_name forward_selection --runs 2