# for batch_size in 32 64
# do
# python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
#     --lr 5e-5 --batch_size $batch_size --max_length 256 --epochs 10\
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --devices 2 \
#     --write_results --runs 1 --save_name tune
# done
# --task_idxes 0 1 2 

python custom_train_alpaca.py --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy fsdp --devices 0 1 2 --runs 1 --precision 16 --accumulate 1 

# python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125M \
#     --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --strategy auto --devices 0 --runs 1 
# --load_model_dir Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
# --project_gradients --project_dimension 200