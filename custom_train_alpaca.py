import argparse
import logging
import os
import wandb

from src.custom.alpaca_data_module import AlpacaDataModule
from src.custom.instruction_data_module import InstructionDataModule
from src.custom.alpaca_model import AlpacaModel
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from pytorch_lightning.strategies import FSDPStrategy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision("high")

def _and_policy(
    module: torch.nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    policies,
) -> bool:

    return not any(not policy(module, recurse, nonwrapped_numel) for policy in policies)

def evaluate(outputs, model, tokenizer):
    loss_dict = defaultdict(list)
    for batch in outputs:
        skills = batch["skills"]
        losses = batch["losses"]
        for j, skill in enumerate(skills):
            loss_dict[skill].append(losses[j])   

    summary = {}
    for skill, losses in loss_dict.items():
        summary[f"loss_{skill}"] = torch.stack(losses).mean().item()

    # Log metrics
    logging.info(summary)
    return summary
    
def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--train_instruction", action="store_true")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD")
    
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    if args.precision == 16:
        args.precision = "bf16"
        print("Setting precision to bf16")

    model_key = args.model_key.replace("/", "-")
    save_name = f"Alpaca_{model_key}" + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_{args.save_name}" if args.save_name else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    for run in range(args.runs):
        if "gpt" in args.model_key or "Llama" in model_key:
            hf_key = args.model_key.replace("_", "-")
            tokenizer = AutoTokenizer.from_pretrained(hf_key)
            model = AutoModelForCausalLM.from_pretrained(hf_key)
            model_type = "decoder"
            append_eos = True
        else:
            raise NotImplementedError(args.model_key)
        
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

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = args.batch_size
        if args.inference_batch_size is None:
            inference_batch_size = batch_size
        else:
            inference_batch_size = args.inference_batch_size
        if args.train_instruction:
            task_idxes = args.task_idxes if args.task_idxes is not None else list(range(1729))
            data_module = InstructionDataModule(tokenizer=tokenizer,
                                task_idxes=task_idxes,
                                batch_size = batch_size,
                                inference_batch_size = inference_batch_size,
                                context_length=args.max_length)
        else:
            # Only load alpaca dataset
            task_idxes = args.task_idxes if args.task_idxes is not None else list(range(38))
            data_module = AlpacaDataModule(tokenizer=tokenizer,
                                data_path="./data/alpaca_data/alpaca_final.pkl",
                                dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                                task_idxes=task_idxes,
                                batch_size = batch_size,
                                inference_batch_size = inference_batch_size,
                                context_length=args.max_length)
            
        use_cpu_offload = args.strategy and "offload" in args.strategy
        lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=use_cpu_offload,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=run, 
                        project_dim=args.project_dimension, gradients_dir=save_name + f"_run_{run}")
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
            lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=run, 
                        project_dim=args.project_dimension, gradients_dir=save_name + f"_run_{run}")
            logging.info(f"Loaded model from {load_model_dir}")

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        data_module.setup(stage="fit")
        task_name = "_".join(data_module.skills[:5])
        default_root_dir = os.path.join("external_lightning_logs", 
                                        ("Instruction__{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                                        (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                        (f"_{args.save_name}" if args.save_name else "") + \
                                        (f"_task_{task_name}") + \
                                        f"_run_{run}"
                                        )
        # remove previous checkpoints
        if args.save_name and os.path.exists(default_root_dir):
            os.system(f"rm -rf {default_root_dir}")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="loss",
            dirpath=default_root_dir,
            filename="epoch_{epoch}",
            save_top_k=(-1 if args.save_every_epoch else 1),
            mode="min",
        )
        if args.strategy == "fsdp":
            def lambda_policy_fn(module):
                if (
                    len(list(module.named_children())) == 0
                    and getattr(module, "weight", None) is not None
                    and module.weight.requires_grad
                ):
                    return True
                return False
            lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
            transformer_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(
                    GPTNeoBlock if "gpt" in model_key else LlamaDecoderLayer
                )
            )
            auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
            strategy = FSDPStrategy(auto_wrap_policy=lambda_policy)
        else:
            strategy = "auto"
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )
        if args.use_wandb:
            run_name = ("Instruction__{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                    (f"_lr_{args.lr}_batch_{args.batch_size}") + \
                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                    (f"_{args.save_name}" if args.save_name else "") +\
                    (f"_task_{task_name}") \
                    + f"_run_{run}"
            wandb.init(project="scalable-mtl", name=run_name)
        if args.epochs > 0:
            trainer.fit(lm, datamodule=data_module)
        if args.use_wandb:
            wandb.finish()

        # evaluate the best checkpoint
        if args.epochs > 0:
            lm = AlpacaModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=model, tokenizer=tokenizer, model_type=model_type,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                    intialize_project_matrix=args.project_gradients, run_seed=run, 
                    project_dim=args.project_dimension, gradients_dir=save_name + f"_run_{run}")

        if args.project_gradients:
            trainer.predict(lm, dataloaders=data_module.train_dataloader())
        summary = trainer.validate(lm, datamodule=data_module)[0]
        logging.info(summary)

        # save indexes 
        if args.write_results and run == 0:
            subset_idxes = task_idxes
            for i, idx in enumerate(subset_idxes):
                result_datapoint = {
                    "Target task": idx,
                    "Data indices": " ".join([str(idx) for idx in subset_idxes])
                }
                result_datapoint["Loss"] = summary[f"loss_{data_module.skills[i]}"]
        
                file_name = os.path.join(file_dir, "results.csv")
                add_result_to_csv(result_datapoint, file_name)
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])
    
    for key in metrics:
        logging.info(f"{key}: {np.mean(metrics[key])} +/- {np.std(metrics[key])}")