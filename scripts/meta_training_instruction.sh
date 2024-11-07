python custom_train_alpaca.py --train_instruction --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --lr 2e-5 --batch_size 128 --max_length 256 --epochs 10\
    --train_lora --lora_rank 128 --lora_alpha 512\
    --strategy auto --devices 0 1 2 3 --runs 1 --precision 16 --accumulate 1