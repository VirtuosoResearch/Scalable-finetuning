python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0

for seed in 0 1 2
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.1 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.06 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "google/gemma-2b" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_google-gemma-2b_lora_r_4_run_0
done


python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --compute_pretrained_outputs --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0

for seed in 0 1 2 
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.1 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.06 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0
python fast_estimate_eval_approximation_alpaca.py \
    --model_key "mistralai/Mistral-7B-v0.3" --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 0 1 2 --strategy auto --save_name fast_approximation --num_batches_gradients 200\
    --load_model_dir Alpaca_mistralai-Mistral-7B-v0.3_lora_r_4_run_0
done
