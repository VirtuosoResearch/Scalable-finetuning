python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --devices 2 --runs 1 \
    --train_sam --sam_rho 0.02 --save_every_epoch 