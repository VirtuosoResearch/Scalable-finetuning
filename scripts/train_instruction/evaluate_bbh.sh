CUDA_VISIBLE_DEVICES=2 python -m eval.bbh.run_eval \
                    --data_dir data/eval/bbh \
                    --save_dir results/bbh/pretrained_TinyLlama \
                    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
                    --tokenizer TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
                    --max_num_examples_per_task 40 \
                    --no_cot

CUDA_VISIBLE_DEVICES=2 python -m eval.bbh.run_eval \
                    --data_dir data/eval/bbh \
                    --save_dir results/bbh/pretrained_TinyLlama_cot \
                    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T\
                    --tokenizer TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
                    --max_num_examples_per_task 40 