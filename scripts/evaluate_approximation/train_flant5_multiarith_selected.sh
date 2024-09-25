for ratio in 95 90 85 80 75 70 65 60 55 50
do
python custom_train.py --dataset_key multiarith --model_key flan_t5_base --train_key ft_cot \
    --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 \
    --data_index_dir "multiarith_ft_cot_ours_ratio_$ratio"\
    --save_name "ours_ratio_$ratio" --runs 2
done