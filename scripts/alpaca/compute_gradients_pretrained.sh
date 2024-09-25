python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-125m --precision 32\
    --compute_pretrained_outputs --batch_size 8 --project_gradients --project_dimension 100 --devices 1 --run 0

# --train_lora --lora_rank 4 --lora_alpha 32 