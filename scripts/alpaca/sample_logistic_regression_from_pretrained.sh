# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125M --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
#     --compute_pretrained_outputs --batch_size 8 --project_gradients --project_dimension 200 --devices 1 --run 0

python fast_estimate_linear_regression_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --precision 32\
    --batch_size 8 --project_dimension 100 --devices 0\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 50 --subset_size 0.5 --scale 0.05

# --train_lora --lora_rank 4 --lora_alpha 32 