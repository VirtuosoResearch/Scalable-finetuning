python fast_estimate_eval_approximation_alpaca.py --train_instruction \
    --model_key TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train_lora --lora_rank 128 --lora_alpha 512 --precision 32\
    --load_model_dir Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_task_add_adversarial_qa_dbert_answer_the_following_q_adversarial_qa_dbert_based_on_adversarial_qa_dbert_generate_question_adversarial_qa_dbert_question_context_answer_run_0/epoch_epoch=0\
    --batch_size 8 --max_length 256 --project_gradients --project_dimension 100\
    --compute_pretrained_outputs --number_of_batches 100000\
    --devices 0 1 --strategy fsdp
