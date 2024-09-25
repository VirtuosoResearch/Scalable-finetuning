# %%
import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from peft import get_peft_model, LoraConfig


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class args:
    model_key = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" # "EleutherAI/gpt-neo-1.3B"
    train_lora = True
    lora_rank = 4
    lora_alpha = 32
    precision = 16

    lr = 5e-5
    weight_decay = 1e-4
    max_length = 512
    use_wandb = False
    load_model_dir = "Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_pairwise_task_add_analyze_arrange_calculate_categorize_run_0/epoch_epoch=8"

print("arguments".upper().center(80, "-"))
print(args)
print("-" * 80)

if args.precision == 16:
    args.precision = "bf16"
    print("Setting precision to bf16")

model_key = args.model_key

if "gpt" in model_key or "Llama" in model_key:
    hf_key = model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    model = AutoModelForCausalLM.from_pretrained(hf_key,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",)
    model_type = "decoder"
    append_eos = True
else:
    raise NotImplementedError(model_key)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# %%
if args.train_lora:
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

data_module = AlpacaDataModule(tokenizer=tokenizer,
                               data_path="./data/alpaca_data/alpaca_final.pkl",
                               dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                               task_idxes=list(range(38)),
                               batch_size = 8,
                               inference_batch_size = 8,
                               context_length=256)
data_module.setup(stage="fit")

lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb)

# %%

load_model_dir = args.load_model_dir
load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
    lm = lm.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type)
    logging.info(f"Loaded model from {load_model_dir}")


# %%

