python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_dsir_ratio_0.9\
    --save_name dsir_ratio_90

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_dsir_ratio_0.8\
    --save_name dsir_ratio_80

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_dsir_ratio_0.7\
    --save_name dsir_ratio_70

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_dsir_ratio_0.6\
    --save_name dsir_ratio_60

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --data_index_dir multiarith_ft_cot_t70_64aug_dsir_ratio_0.5\
    --save_name dsir_ratio_50