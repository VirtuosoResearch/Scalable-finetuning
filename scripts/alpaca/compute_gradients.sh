# python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-1.3B \
#     --lr 5e-5 --batch_size 4 --max_length 256 --epochs 0\
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --strategy fsdp --devices 0 1 2 --runs 1 --precision 16 --accumulate 1\
#     --load_model_dir Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9\
#     --project_gradients --project_dimension 200

# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key EleutherAI/gpt-neo-1.3B --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
#     --batch_size 4 --project_gradients --project_dimension 1000\
#     --scale 0.1 --seed 0 --compute_pretrained_outputs \
#     --load_model_dir Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9

for dim in 50 100 200 400
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125M --train_lora --lora_rank 4 --lora_alpha 32 --precision 32\
    --compute_pretrained_outputs --batch_size 16 --project_gradients --project_dimension $dim --devices 0\
    --load_model_dir Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_pairwise_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=8
done
