
python custom_train_alpaca.py --model_key "google/gemma-2b"\
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 5\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 1 --runs 1 --precision "bf16-true" --accumulate 4 --save_every_epoch --save_name quantized_meta_initialization\
    --use_qlora --optimizer "paged_adamw_32bit"

# python custom_train_alpaca.py --model_key "google/gemma-2b" \
#     --lr 5e-5 --batch_size 4 --max_length 256 --epochs 10\
#     --train_lora --lora_rank 4 --lora_alpha 32\
#     --strategy auto --devices 0 1 2 --runs 1 --precision "bf16-true" --accumulate 1 --save_every_epoch --save_name meta_initialization_alpaca

python custom_train_alpaca.py --model_key "../llama/llama-3/Meta-Llama-3-8B-hf"\
    --lr 5e-5 --batch_size 4 --max_length 256 --epochs 5\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 1 --runs 1 --precision "bf16-true" --accumulate 4 --save_every_epoch --save_name quantized_meta_initialization\
    --use_qlora --optimizer "paged_adamw_32bit"