# python custom_train.py --dataset_key gsm8k --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 4 --lora_alpha 32 --runs 5

# python custom_train.py --dataset_key gsm8k --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 8 --lora_alpha 64 --runs 3

# python custom_train.py --dataset_key gsm8k --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 32 --lora_alpha 256 --runs 3

# python custom_train.py --dataset_key gsm8k --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 64 --lora_alpha 512 --runs 3

# for wd in 1e-4 1e-3 1e-2
# do
# python custom_train.py --dataset_key gsm8k --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 16 --lora_alpha 128 --runs 1 --save_name new --weight_decay $wd

# python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 2 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 4 --lora_alpha 32 --runs 1 --save_name new --weight_decay $wd
# done