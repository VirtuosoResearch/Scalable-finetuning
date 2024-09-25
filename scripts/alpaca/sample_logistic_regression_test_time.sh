# python fast_estimate_eval_approximation_alpaca.py \
#     --model_key gpt2 --precision "bf16-true"\
#     --compute_pretrained_outputs --batch_size 16 --project_gradients --project_dimension 100 --devices 1 --run 0\
#     --load_model_dir Alpaca_gpt2_test_time_run_0/epoch_epoch=0 --save_name test_time --num_batches_gradients 100 --strategy "auto"

python fast_estimate_eval_approximation_alpaca.py \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --precision "bf16-true" \
    --train_lora --lora_rank 4 --lora_alpha 32\
    --compute_pretrained_outputs --batch_size 4 --project_gradients --project_dimension 100 --devices 0 1 2 --run 0\
    --save_name test_time --num_batches_gradients 100
# --load_model_dir Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_test_time_run_0/epoch_epoch=0 


python fast_estimate_eval_approximation_alpaca.py \
    --model_key EleutherAI/gpt-neo-1.3B  --precision "bf16-true"  \
    --train_lora --lora_rank 4 --lora_alpha 32\
    --compute_pretrained_outputs --batch_size 4 --project_gradients --project_dimension 100 --devices 0 1 2 --run 0\
    --save_name test_time --num_batches_gradients 100
# --load_model_dir Alpaca_EleutherAI-gpt-neo-1.3B_lora_r_4_test_time_run_0/epoch_epoch=0 

python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf  --precision "bf16-true" \
    --train_lora --lora_rank 4 --lora_alpha 32\
    --compute_pretrained_outputs --batch_size 2 --project_gradients --project_dimension 100 --devices 0 1 2 --run 0\
    --save_name test_time --num_batches_gradients 100
# --load_model_dir Alpaca_-llama-llama-3-Meta-Llama-3-8B-hf_lora_r_4_test_time_run_0/epoch_epoch=0 