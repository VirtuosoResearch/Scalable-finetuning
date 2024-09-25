# from 0 to 38
for idx in {0..37}
do
python custom_train_alpaca.py --model_key EleutherAI/gpt-neo-125m \
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32 --task_idxes $idx\
    --devices 0 \
    --write_results --runs 1 --save_name single
done