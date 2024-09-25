# python custom_train.py --dataset_key single_eq --model_key flan_t5_base --train_key ft_cot \
#     --preset_key ft_cot_t70_8aug --devices 0 --batch_size 8 --inference_batch_size 32


python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 0 --batch_size 8 --inference_batch_size 32

python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 0 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32