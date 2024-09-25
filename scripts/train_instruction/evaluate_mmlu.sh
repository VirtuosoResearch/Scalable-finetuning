# CUDA_VISIBLE_DEVICES=1 python -m eval.mmlu.run_eval \
#         --ntrain 0 \
#         --data_dir data/eval/mmlu \
#         --save_dir results/mmlu/pretrained_TinyLlama \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --eval_batch_size 4 \
#         --load_in_8bit

CUDA_VISIBLE_DEVICES=1 python -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu \
        --save_dir results/mmlu/pretrained_TinyLlama \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --eval_batch_size 4 \
        --load_in_8bit \
        --adapter_path exported_model/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_epoch_0

CUDA_VISIBLE_DEVICES=1 python -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir data/eval/mmlu \
        --save_dir results/mmlu/pretrained_TinyLlama \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --eval_batch_size 4 \
        --load_in_8bit \
        --adapter_path exported_model/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_epoch_1