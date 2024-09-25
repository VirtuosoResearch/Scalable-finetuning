# python custom_train_alpaca.py --model_key gpt2 \
#     --lr 5e-5 --batch_size 16 --max_length 256 --epochs 1\
#     --devices 0 1 2 --runs 1 --save_name test_time --precision "bf16-true" 

# python custom_train_alpaca.py --model_key flan_t5_base \
#     --lr 5e-5 --batch_size 16 --max_length 256 --epochs 1\
#     --devices 0 1 2 --runs 1 --save_name test_time

# python custom_train_alpaca.py --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --train_lora --lora_rank 4 --lora_alpha 32 \
#     --lr 5e-5 --batch_size 4 --accumulate 4 --max_length 256 --epochs 1\
#     --devices 0 1 2 --runs 1 --save_name test_time --precision "bf16-true" 

# python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-1.3B \
#     --train_lora --lora_rank 4 --lora_alpha 32 \
#     --lr 5e-5 --batch_size 4 --accumulate 4 --max_length 256 --epochs 1\
#     --devices 0 1 2 --runs 1 --save_name test_time --precision "bf16-true" 

# python custom_train_alpaca.py --model_key ../llama/llama-3/Meta-Llama-3-8B-hf \
#     --train_lora --lora_rank 4 --lora_alpha 32 \
#     --lr 5e-5 --batch_size 2 --accumulate 8 --max_length 256 --epochs 1\
#     --devices 0 1 2 --runs 1 --save_name test_time --precision "bf16-true" 
