# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key gpt2 --precision 32\
#     --compute_pretrained_outputs --batch_size 8 --project_gradients --project_dimension 100 --devices 0 --run 0\
#     --load_model_dir Alpaca_gpt2_meta_initialization_run_0/epoch_epoch=1 --save_name sample_logistic_regression

python fast_estimate_linear_regression_alpaca.py \
    --model_key gpt2 --precision 32\
    --batch_size 8 --project_gradients --project_dimension 100 --devices 0 1 2\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 100 --subset_size 0.5\
    --load_model_dir Alpaca_gpt2_meta_initialization_run_0/epoch_epoch=10 --save_name sample_logistic_regression

# python fast_estimate_linear_regression_alpaca.py \
#     --model_key gpt2 --precision 32\
#     --batch_size 4 --project_gradients --project_dimension 100 --devices 0 1 2\
#     --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 100 --subset_size 0.5\
#     --load_model_dir Alpaca_gpt2_meta_initialization_run_0/epoch_epoch=10 --save_name test_time