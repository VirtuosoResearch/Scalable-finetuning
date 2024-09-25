# python fast_estimate_compute_gradients.py --dataset_key multiarith --model_key flan_t5_base\
#     --batch_size 8 --preset_key ft_cot \
#     --load_model_dir flan_t5_base_multiarith_ft_cot_lora_r_4_run_1/epoch_epoch=19\
#     --project_dim 200 --create_projection --run 1\
#     --train_lora --lora_rank 4 --lora_alpha 32 --device 2

# python fast_estimate_compute_gradients.py --dataset_key multiarith --model_key flan_t5_base\
#     --batch_size 8 --preset_key ft_cot \
#     --load_model_dir flan_t5_base_multiarith_ft_cot_lora_r_4_run_2/epoch_epoch=19\
#     --project_dim 200 --create_projection --run 2\
#     --train_lora --lora_rank 4 --lora_alpha 32 --device 2

python fast_estimate_compute_gradients.py --dataset_key multiarith --model_key flan_t5_base\
    --batch_size 8 --preset_key ft_cot \
    --load_model_dir flan_t5_base_multiarith_ft_cot_lora_r_4_run_3/epoch_epoch=19\
    --project_dim 200 --create_projection --run 3\
    --train_lora --lora_rank 4 --lora_alpha 32 --device 1

# python fast_estimate_compute_gradients.py --dataset_key multiarith --model_key flan_t5_base\
#     --batch_size 8 --preset_key ft_cot \
#     --load_model_dir flan_t5_base_multiarith_ft_cot_lora_r_4_run_4/epoch_epoch=19\
#     --project_dim 200 --create_projection --run 4\
#     --train_lora --lora_rank 4 --lora_alpha 32 --device 2