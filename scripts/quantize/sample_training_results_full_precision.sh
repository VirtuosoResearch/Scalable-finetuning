task_index_list=("1 5 8 9 10 14 15 16 19 20 21 23 25 28 29 30 34 35 36"
"0 1 3 4 8 9 10 11 12 16 21 23 25 26 28 31 34 35 36"
"0 7 8 12 13 16 17 19 20 22 23 25 27 28 29 30 32 34 36"
"1 2 5 6 7 9 10 12 13 14 22 23 25 27 29 31 34 35 37"
"0 1 3 5 6 8 9 10 11 15 16 18 20 24 25 31 32 35 37"
"0 5 7 10 16 17 18 19 20 23 24 27 29 30 31 32 34 36 37"
"1 2 6 7 9 10 11 19 20 22 23 25 27 28 29 31 33 34 36")

for task_idxes in "${task_index_list[@]}"
do
python custom_train_alpaca.py --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 5\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 2 --runs 1 --precision "bf16-true" --accumulate 1 --save_name sample_full_precision_training\
    --task_idxes $task_idxes --write_results
done

# "4 5 6 7 8 10 11 15 17 18 22 23 25 27 28 29 30 33 34"
# "0 3 6 7 8 9 17 19 20 21 23 26 28 29 33 34 35 36 37"
# "0 2 4 5 7 8 12 13 19 21 22 23 28 29 30 31 32 33 34"
