python custom_train_alpaca.py --model_key "google/gemma-2b" \
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 20\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 0 1 2 --runs 1 --precision "bf16-true" --accumulate 1 --downsample 400

python custom_train_alpaca.py --model_key "mistralai/Mistral-7B-v0.3" \
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 20\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 0 1 2 --runs 1 --precision "bf16-true" --accumulate 1 --downsample 400