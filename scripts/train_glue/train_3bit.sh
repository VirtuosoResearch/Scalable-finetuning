for task_name in "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc"
do
python custom_train_glue.py --task_name $task_name --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 3 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name meta_train_3bit --epochs 10 --use_3bit --optimizer "paged_adamw_32bit" --precision "32" 
done