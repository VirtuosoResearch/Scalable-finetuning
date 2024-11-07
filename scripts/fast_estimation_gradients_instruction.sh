python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name test

# the following are the examples for evaluating the approximation errors
for seed in 0 1 2 3 4
do

python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.10 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.06 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 0 1 2 --strategy auto
python fast_estimate_eval_approximation_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision "bf16-true"\
    --load_model_dir $model_checkpoint\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 0 1 2 --strategy auto
done
