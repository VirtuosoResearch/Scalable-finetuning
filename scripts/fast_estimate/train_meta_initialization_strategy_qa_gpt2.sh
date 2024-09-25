python custom_train.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --devices 2 --batch_size 8 --inference_batch_size 32 --runs 1\
    --save_name meta_train --epochs 5