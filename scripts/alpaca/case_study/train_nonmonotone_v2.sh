python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 \
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 \
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 \
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5\
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5 31 \
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5 31 35 \
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5 31 35 36\
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5 31 35 36 16\
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 30 2 27 18 6 5 31 35 36 16\
    --devices 1 \
    --write_results --runs 1 --save_name higherorder --use_wandb