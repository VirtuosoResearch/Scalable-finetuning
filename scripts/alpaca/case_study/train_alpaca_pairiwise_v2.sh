for idx_1 in 30 33 34 35
do
for idx_2 in {0..37}
do
python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes $idx_1 $idx_2\
    --devices 2 \
    --write_results --runs 1 --save_name pairwise --use_wandb
done
done