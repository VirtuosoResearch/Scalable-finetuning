python custom_train_alpaca.py --model_key ../llama/llama-3/Meta-Llama-3-8B-hf  \
    --lr 5e-5 --batch_size 1 --max_length 8 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 0 1 2 --runs 1 --accumulate 1 --precision "bf16-true" 
# ../llama/llama-3/Meta-Llama-3-8B-hf 