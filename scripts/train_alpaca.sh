python custom_train_alpaca.py --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy fsdp --devices 0 1 2 --runs 1 --precision 16 --accumulate 1 
