# %%
import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from peft import get_peft_model, LoraConfig


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class args:
    model_key =  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #  # "google/gemma-2b" # "EleutherAI/gpt-neo-1.3B" # "mistralai/Mistral-7B-v0.3" #
    train_lora = True
    lora_rank = 4
    lora_alpha = 32

    lr = 5e-5
    weight_decay = 1e-4
    max_length = 256
    use_wandb = False
    load_model_dir =  "Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=9" # "Alpaca_google-gemma-2b_lora_r_4_run_0/epoch_epoch=0" #
    save_model_dir = "Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_task_add_analyze_arrange_calculate_categorize_run_0_epoch_9"

    use_qlora = False
    device = 2


print("arguments".upper().center(80, "-"))
print(args)
print("-" * 80)

model_key = args.model_key

if "gpt" in args.model_key or "Llama" in model_key \
    or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
            )
        model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                                                     device_map={"": args.device}) 
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
elif "flan" in model_key:
    hf_key = "google/{}".format(model_key.replace("_", "-"))
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
    tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
    model_type = "encoder_decoder"
    append_eos = False  # t5 tokenizers already append eos
else:
    raise NotImplementedError(args.model_key)

if args.train_lora:
    if args.model_key == "gpt2":
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
    elif args.model_key == "flan":
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q", "k", "v"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
    else:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb)

# %%
load_model_dir = args.load_model_dir
load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
    lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt",  map_location=f"cuda:{args.device}", strict=False, model=model, tokenizer=tokenizer, model_type=model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length)
    logging.info(f"Loaded model from {load_model_dir}")
# %%
def get_state_dict(model):
    state_dict = model.state_dict()
    returned_state_dict = {}
    for k in state_dict.keys():
        if "lora" in k: 
            returned_state_dict[k] = state_dict[k].cpu().clone()
    return returned_state_dict

list(get_state_dict(lm.model).keys())
# %%
torch.save(get_state_dict(lm.model), f"./exported_model/{args.save_model_dir}.pt")

# %%
torch.load(f"./exported_model/{args.save_model_dir}.pt")