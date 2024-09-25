python sample_train_results_alpaca.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --precision 32 --lr 5e-5\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 0 1 2 --subset_num 100\
    --save_name fully_finetuned

# python custom_train_alpaca.py --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --lr 5e-5 --batch_size 16 --max_length 256 --epochs 1 --precision 32\
#     --devices 0 1 2 --runs 1 --task_idxes 0 --save_name test

# python custom_train_alpaca.py --model_key ../llama/llama-3/Meta-Llama-3-8B-hf \
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --lr 5e-5 --batch_size 1 --max_length 256 --epochs 1 --precision "bf16-true" \
#     --devices 0 1 2 --runs 1 --task_idxes 0 --save_name test