python sample_train_results_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --precision 32 --lr 1.8e-3 \
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 2 --subset_num 50 --epochs 1 --save_name epoch_1

# --lora_rank 4 --lora_alpha 32
# python sample_train_results_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
#     --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 1 --subset_num 5 --epochs 2 --save_name epoch_2

# python sample_train_results_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
#     --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 1 --subset_num 5 --epochs 3 --save_name epoch_3