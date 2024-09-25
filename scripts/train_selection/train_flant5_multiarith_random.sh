python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_random_ratio_0.9

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_random_ratio_0.8

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_random_ratio_0.7

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_random_ratio_0.6

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_random_ratio_0.5