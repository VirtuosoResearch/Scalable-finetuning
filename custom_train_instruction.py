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
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation
import pandas as pd
from collections import defaultdict
import time

from adapters import AutoAdapterModel,list_adapters, BnConfig

logging.basicConfig(level=logging.INFO)

# torch.set_float32_matmul_precision("high")

def _and_policy(
    module: torch.nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    policies,
) -> bool:
    """
    A policy that wraps ``module`` if all policy in the passed in iterable of
    ``policies`` returns ``True``.
    """
    return not any(not policy(module, recurse, nonwrapped_numel) for policy in policies)

def evaluate(outputs, model, tokenizer):
    """
    Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
    log validation metrics.

    Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
    """
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

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
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
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)    
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_instruction", action="store_true") # if use the larger instruction dataset
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_qlora", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    # SAM or freezing head parameters, not frequentely used
    parser.add_argument('--train_sam', action="store_true")
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--sam_adaptive', action="store_true")
    parser.add_argument('--sam_unnormalize', action="store_true")

    parser.add_argument("--freeze_head", action="store_true")
    
    parser.add_argument("--use_adapter", action="store_true")
    
    # Additional operations 
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = ("Instruction__{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_{args.save_name}" if args.save_name else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    for run in range(args.runs):
        model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

        if args.freeze_head:
            model.lm_head.weight.requires_grad = False
            model.transformer.wte.weight.requires_grad = False
            model.transformer.wpe.weight.requires_grad = False

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
                                context_length=args.max_length,
                                downsample=args.downsample,
                                model_type=model_type)
        data_module.setup(stage="fit")
        
        use_cpu_offload = args.strategy and "offload" in args.strategy
        lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=use_cpu_offload,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=run, 
                        train_sam=args.train_sam, sam_rho=args.sam_rho, sam_adaptive=args.sam_adaptive, sam_unnormalize=args.sam_unnormalize,
                        project_dim=args.project_dimension, gradient_dir=save_name + f"_run_{run}", use_sgd=False,
                        optimizer=args.optimizer)
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
            lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=run, 
                        train_sam=args.train_sam, sam_rho=args.sam_rho, sam_adaptive=args.sam_adaptive, sam_unnormalize=args.sam_unnormalize,
                        project_dim=args.project_dimension, gradient_dir=save_name + f"_run_{run}", use_sgd=False)
            logging.info(f"Loaded model from {load_model_dir}")

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        data_module.setup(stage="fit")
        task_name = "_".join(data_module.skills[:5])[:100]
        default_root_dir = os.path.join("external_lightning_logs", 
                                        ("Instruction_{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                                        (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                        (f"_{args.save_name}" if args.save_name else "") + \
                                        (f"_sam_rho_{args.sam_rho}" if args.train_sam else "") + \
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
            ''' Deprecated '''
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
            ''' Deprecated '''
        else:
            strategy = "auto"


        
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )
        if args.train_lora:
            if not os.path.exists(default_root_dir):
                os.makedirs(default_root_dir)
            model_path = default_root_dir + "/initial_weights.pt"
            state_dict = model.state_dict()
            state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, model_path)
        if args.use_wandb:
            run_name = ("Instruction__{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                    (f"_lr_{args.lr}_batch_{args.batch_size}") + \
                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                    (f"_{args.save_name}" if args.save_name else "") +\
                    (f"_task_{task_name}") +\
                    f"_run_{run}"
            wandb.init(project="scalable-mtl", name=run_name)

        start_time = time.time()
        if args.epochs > 0:
            trainer.fit(lm, datamodule=data_module)
        end_time = time.time()
        print(f"Training time: {end_time - start_time}")
        if args.use_wandb:
            wandb.finish()

        if args.train_lora:
            from lightning_fabric.utilities.cloud_io import _load as pl_load
            checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
            state_dict = checkpoint["state_dict"]
            state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))

        # evaluate the best checkpoint
        start_time = time.time()
        if args.epochs > 0:
            if args.use_qlora:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k: v for k, v in state_dict.items() if "lora" in k}       
                         
                model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
                model.load_state_dict(state_dict, strict=False)
                lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=use_cpu_offload,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=run, 
                        train_sam=args.train_sam, sam_rho=args.sam_rho, sam_adaptive=args.sam_adaptive, sam_unnormalize=args.sam_unnormalize,
                        project_dim=args.project_dimension, gradient_dir=save_name + f"_run_{run}", use_sgd=False,
                        optimizer=args.optimizer)
                
                summary = trainer.validate(lm, datamodule=data_module)[0]
            else:
                summary = trainer.validate(lm, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)[0]
            logging.info(summary)
        else:
            summary = trainer.validate(lm, datamodule=data_module)[0]
            logging.info(summary)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time}")

        # save indexes 
        if args.write_results and run == 0:
            subset_idxes = task_idxes
            for i, idx in enumerate(list(range(38))):
                result_datapoint = {
                    "Data indices": " ".join([str(idx) for idx in subset_idxes])
                }
                for key, val in summary.items():
                    result_datapoint[key] = val
                file_name = os.path.join(file_dir, "results.csv")
                add_result_to_csv(result_datapoint, file_name)
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])

        # delete the whole model checkpoint and only keep the lora parameters
        if args.train_lora or args.train_adapter:
            os.system(f"rm {checkpoint_callback.best_model_path}")
    
    for key in metrics:
        logging.info(f"{key}: {np.mean(metrics[key])} +/- {np.std(metrics[key])}")
