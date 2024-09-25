accelerate launch --use_fsdp --gpu_ids 0,1,2 \
    -m lm_eval --model hf \
    --model_args pretrained=EleutherAI/GPT-Neo-1.3B \
    --tasks truthfulqa --batch_size 4 