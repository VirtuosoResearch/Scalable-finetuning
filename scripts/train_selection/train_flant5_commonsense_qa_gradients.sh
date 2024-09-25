python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_95\
    --save_name gradients_ratio_95 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_90\
    --save_name gradients_ratio_90 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_85\
    --save_name gradients_ratio_85 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_80\
    --save_name gradients_ratio_80 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_75\
    --save_name gradients_ratio_75 --runs 2

python custom_train.py --dataset_key commonsense_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --data_index_dir commonsense_qa_ft_cot_gradients_ratio_70\
    --save_name gradients_ratio_70 --runs 2