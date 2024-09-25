CUDA_VISIBLE_DEVICES=0 python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/pretrained_TinyLlama \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
    --load_in_8bit\
    --eval_batch_size 256