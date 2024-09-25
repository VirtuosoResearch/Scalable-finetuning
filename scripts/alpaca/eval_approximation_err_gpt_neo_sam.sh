# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 100\
#     --scale 0.1 --seed 0 --compute_pretrained_outputs --precision 16 --devices 2 --strategy auto\
#     --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
#     --save_name sam

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.2 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam

python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.16 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam

python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.12 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam

python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam

python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam

python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 2 --strategy auto\
    --load_model_dir ./Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_task_add_analyze_arrange_calculate_categorize_sam_rho_0.02_run_0/epoch_epoch=8\
    --save_name sam
done


# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-125m --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 100\
#     --scale 0.1 --seed 0 --devices 2 --strategy auto