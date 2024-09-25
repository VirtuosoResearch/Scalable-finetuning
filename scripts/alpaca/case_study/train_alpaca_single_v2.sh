for idx in 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37
do
python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes $idx\
    --devices 2 \
    --write_results --runs 1 --save_name tune
done