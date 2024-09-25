python sample_train_results_alpaca.py \
    --model_key gpt2 --precision 32\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 1\
    --save_name sampled_train --subset_num 100 --epochs 10