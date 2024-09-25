

# for rank in 4 8 16 32 64
# do
# python custom_train_glue.py --task_name "rte" --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 2 --lr 5e-5\
#     --train_lora --lora_rank $rank --lora_alpha $((rank*8)) \
#     --save_name meta_train --epochs 10 
# done

for batch_size in 16 32
do
python custom_train_glue.py --task_name "rte" --model_key "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --devices 1 --batch_size $batch_size --inference_batch_size $batch_size --max_length 128 --runs 2 --lr 5e-5\
    --train_lora --lora_rank 64 --lora_alpha 512 \
    --save_name meta_train --epochs 10 
done


# for lr in 1e-4 2e-4 5e-4 # 1e-5 2e-5 5e-5
# do
# python custom_train_glue.py --task_name "rte" --model_key  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 128 --runs 2 --lr $lr\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --save_name meta_train --epochs 10 
# done