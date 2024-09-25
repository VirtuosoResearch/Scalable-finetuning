python custom_train.py --dataset_key strategy_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --runs 1 --epochs 1

python custom_train.py --dataset_key strategy_qa --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 1 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 16 --lora_alpha 128 --runs 1 --epochs 0

python fast_estimate_compute_gradients.py --dataset_key strategy_qa --model_key flan_t5_base --preset_key ft_cot_t70_64aug\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --run 0 --project_dim 100 --device 2 --load_model_dir flan_t5_base_strategy_qa_ft_cot_t70_64aug_lora_r_16_new_run_0/epoch_epoch=17