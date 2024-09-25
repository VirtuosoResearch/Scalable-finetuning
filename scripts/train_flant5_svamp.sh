# python custom_train.py --dataset_key svamp --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 4 --lora_alpha 32 --runs 1 --save_name new

# python custom_train.py --dataset_key svamp --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 8 --lora_alpha 64 --runs 1 --save_name new

python custom_train.py --dataset_key svamp --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 0 1 --strategy fsdp --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --runs 1 --save_name new

# python custom_train.py --dataset_key svamp --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 0 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 32 --lora_alpha 256 --runs 1 --save_name new

# python custom_train.py --dataset_key svamp --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
#     --train_lora --lora_rank 4 --lora_alpha 32 --runs 3