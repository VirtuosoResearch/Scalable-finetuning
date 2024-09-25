python fast_estimate_linear_regression_alpaca.py \
    --model_key EleutherAI/gpt-neo-125M --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
    --batch_size 16 --project_gradients --project_dimension 200 --devices 2\
    --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_pairwise_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=8\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 1000 --subset_size 0.5

python fast_estimate_linear_regression_alpaca.py \
    --model_key EleutherAI/gpt-neo-125M --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
    --batch_size 16 --project_gradients --project_dimension 400 --devices 2\
    --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_pairwise_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=8\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 1000 --subset_size 0.5