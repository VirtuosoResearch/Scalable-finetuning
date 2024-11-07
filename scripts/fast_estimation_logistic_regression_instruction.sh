python fast_estimate_linear_regression_instruction.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision 32\
    --batch_size 16 --project_gradients --project_dimension 400 --devices 2\
    --load_model_dir $model_checkpoint\
     --number_of_subsets 1000 --subset_size 0.5