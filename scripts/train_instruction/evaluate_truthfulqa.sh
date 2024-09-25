# CUDA_VISIBLE_DEVICES=0 python -m eval.truthfulqa.run_eval \
#         --data_dir data/eval/truthfulqa \
#         --save_dir results/trutufulqa/pretrained_TinyLLama \
#         --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
#         --metrics mc \
#         --preset qa \
#         --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#         --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#         --eval_batch_size 20 \
#         --load_in_8bit

CUDA_VISIBLE_DEVICES=1 python -m eval.truthfulqa.run_eval \
        --data_dir data/eval/truthfulqa \
        --save_dir results/trutufulqa/fine_tuned_TinyLLama_epoch_0 \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --metrics mc \
        --preset qa \
        --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
        --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
        --eval_batch_size 20 \
        --load_in_8bit\
        --adapter_path exported_model/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_epoch_0

CUDA_VISIBLE_DEVICES=0 python -m eval.truthfulqa.run_eval \
        --data_dir data/eval/truthfulqa \
        --save_dir results/trutufulqa/fine_tuned_TinyLLama_epoch_0 \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
        --metrics mc \
        --preset qa \
        --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
        --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
        --eval_batch_size 20 \
        --load_in_8bit\
        --adapter_path exported_model/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_epoch_1
