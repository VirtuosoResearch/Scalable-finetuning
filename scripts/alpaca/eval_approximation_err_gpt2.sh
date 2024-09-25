python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.1 --seed 0 --compute_pretrained_outputs --precision 16 --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.2 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.16 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.12 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.08 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.04 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2

python fast_estimate_eval_approximation_alpaca.py \
    --model_key gpt2 --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 200\
    --scale 0.02 --seed $seed --devices 2 --strategy auto\
    --load_model_dir Alpaca_gpt2_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
    --save_name gpt2
done
