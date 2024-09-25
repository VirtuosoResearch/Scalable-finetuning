python custom_train_alpaca.py --model_key bigscience/bloomz-560m\
    --lr 5e-5 --batch_size 16 --max_length 256 --epochs 10\
    --strategy auto --devices 1 --runs 1 --precision "bf16-true" --accumulate 1 --save_name meta_train