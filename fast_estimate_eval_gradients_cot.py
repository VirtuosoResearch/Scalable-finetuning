import argparse
import logging
import os

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

from src.custom.model import Model
from torch.utils.data import DataLoader
import numpy as np

import time

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def get_trainable_parameters(model, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm"]):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any([key in name for key in removing_keys]):
            continue
        params.append(param)
    return params

def initialize_model(args):
    model_key = args.model_key
    if "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    elif "t5" in model_key:
        hf_key = model_key.replace("_", "-")
        model = T5ForConditionalGeneration.from_pretrained(hf_key)
        tokenizer = T5TokenizerFast.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False
    elif "gpt" in model_key or "Llama" in model_key \
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
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) #
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(model_key)
    
    if args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
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
    return model, tokenizer, hf_key, model_type, append_eos

def main(args):
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    if args.precision == 16:
        args.precision = "bf16"
        print("Setting precision to bf16")

    dataset_key = args.dataset_key
    model_key = args.model_key
    train_key = args.train_key

    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

    if "ft_cot" in args.preset_key:
        completion_key = "ft_cot"
    elif args.preset_key == "ft":
        completion_key = "ft"
    elif args.preset_key == "fs_cot":
        raise NotImplementedError("We don't train models on fs_cot")
    else:
        raise NotImplementedError(args.preset_key)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    if args.inference_batch_size is None:
        inference_batch_size = batch_size
    else:
        inference_batch_size = args.inference_batch_size
    data_module = DataModule(dataset_key, args.preset_key, tokenizer, model_type, batch_size=batch_size,
                                inference_batch_size=inference_batch_size, num_workers=8, append_eos=append_eos)

    data_module.setup("fit")
    train_loader = DataLoader(
                data_module.train_dataset,
                batch_size=data_module.batch_size,
                num_workers=data_module.num_workers,
                shuffle=False)
    test_loader = DataLoader(
                data_module.test_dataset,
                batch_size=data_module.batch_size,
                num_workers=data_module.num_workers,
                shuffle=False)

    cm = CompletionMetadata(model_key, completion_key, dataset_key, prediction_template=data_module.prediction_template)
    lm = Model(model, tokenizer, model_type, completion_metadata=cm, truncate_early=False)
    load_model_dir = args.load_model_dir

    # if load_model_dir is not None:
    #     load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
    #     lm = Model.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
    #                                     completion_metadata=cm, truncate_early=False)
    if args.load_model_dir is not None:
        load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
        if os.path.exists(load_model_dir):
            state_dict = torch.load(load_model_dir, map_location=lm.model.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from checkpoint from ", load_model_dir)
    device = torch.device(f"cuda:{args.devices[0]}")


    gradients_dir = f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/run_{args.run}" if args.load_model_dir is not None else \
        f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}_pretrained/run_{args.run}"

    if not os.path.exists(gradients_dir):
        os.makedirs(gradients_dir)

    gradient_dim = 0
    remove_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm"]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any([key in name for key in remove_keys]):
                continue
            gradient_dim += param.numel()

    np.random.seed(args.run)
    project_dim = args.project_dim
    project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
    project_matrix *= 1 / np.sqrt(project_dim)

    lm = lm.to(device)
    lm.model.eval()
    # Save gradients
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(lm.device) for k, v in batch.items()}
        # output = lm.training_step(batch, 0)
        
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if lm.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        logits = lm.model(**kwargs)["logits"]

        # get the gradient of the output
        labels = kwargs["labels"]
        if lm.model_type == "decoder":
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        gradients = []
        for i in range(len(labels)):
            tmp_mask = labels[i] != -100
            tmp_logits = logits[i][tmp_mask]
            tmp_probs = torch.softmax(tmp_logits, dim=-1)
            tmp_labels = labels[i][tmp_mask]

            tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels] - 1e-3
            tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs+1e-10))
            tmp_loss = tmp_outputs.mean()

            tmp_gradients = torch.autograd.grad(tmp_loss, get_trainable_parameters(lm.model), retain_graph=True, create_graph=False)
            tmp_gradients = torch.cat([gradient.view(-1) for gradient in tmp_gradients]).cpu().numpy() # flatten gradients
            tmp_gradients = (tmp_gradients.reshape(1, -1) @ project_matrix).flatten()
            gradients.append(tmp_gradients)

        np.save(f"{gradients_dir}/train_batch_{batch_idx}_gradients.npy", gradients)
        if batch_idx >= 1000:
            break
    end_time = time.time()
    print(f"Time taken for train gradients: {end_time - start_time}")

    # for batch_idx, batch in enumerate(test_loader):
    #     batch = {k: v.to(lm.device) for k, v in batch.items()}
        
    #     kwargs = {
    #         "input_ids": batch["input_ids"],
    #         "attention_mask": batch["attention_mask"],
    #         "labels": batch["labels"],
    #     }
    #     if lm.model_type == "encoder_decoder":
    #         kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
    #     logits = lm.model(**kwargs)["logits"]
    #     if lm.model_type == "decoder":
    #         logits = logits[..., :-1, :].contiguous()
    #         labels = labels[..., 1:].contiguous()

    #     # get the gradient of the output
    #     labels = kwargs["labels"]
    #     gradients = []
    #     for i in range(len(labels)):
    #         tmp_mask = labels[i] != -100
    #         tmp_logits = logits[i][tmp_mask]
    #         tmp_probs = torch.softmax(tmp_logits, dim=-1)
    #         tmp_labels = labels[i][tmp_mask]

    #         tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels] - 1e-3
    #         tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs+1e-10))
    #         tmp_loss = tmp_outputs.mean()

    #         tmp_gradients = torch.autograd.grad(tmp_loss, get_trainable_parameters(lm.model), retain_graph=True, create_graph=False)
    #         tmp_gradients = torch.cat([gradient.view(-1) for gradient in tmp_gradients]).cpu().numpy() # flatten gradients
    #         tmp_gradients = (tmp_gradients.reshape(1, -1) @ project_matrix).flatten()
    #         gradients.append(tmp_gradients)

    #     np.save(f"{gradients_dir}/test_batch_{batch_idx}_gradients.npy", gradients)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="multiarith")
    parser.add_argument("--model_key", type=str, default="flan_t5_base")
    parser.add_argument("--train_key", type=str, default="ft_cot")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preset_key", type=str, default="ft_cot_t70_64aug")
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disable_checkpointing", action="store_true")

    parser.add_argument("--load_model_dir", type=str, default=None)

    # projections
    parser.add_argument("--project_dim", type=int, default=200)
    parser.add_argument("--run", type=int, default=0)

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_qlora", action="store_true")

    args = parser.parse_args()
    main(args)