python custom_train.py --dataset_key strategy_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 2 --batch_size 8 --inference_batch_size 32 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 --runs 2