python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 10 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 10 12 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 10 12 32 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 10 12 32 22\
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb

python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes 2 27 30 33 20 23 21 10 12 32 22 28 \
    --devices 0 \
    --write_results --runs 1 --save_name higherorder --use_wandb
