import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from peft import get_peft_model, LoraConfig

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

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

def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings"]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
            # new_state_dict[key] = state_dict[key].clone()
        else:
            new_state_dict[key] = state_dict[key].clone().to(device) + \
                torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
    return new_state_dict

def compute_norm(state_dict, use_lora = True, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings"]):
    norm = 0
    for key, val in state_dict.items():
        if use_lora:
            if "lora" in key:
                norm += val.clone().square().sum().item()
        else:
            if any([rkey in key for rkey in removing_keys]):
                    continue
            norm += val.clone().square().sum().item()
    return np.math.sqrt(norm)

def evaluate_subset(args, trainer, lm, data_module, data_idxes, state_dict, projection_matrix, scale, gradient_dir):
    # collect gradients for the subset
    gradients = []
    for idx in data_idxes:
        gradient_file_idx = idx // args.batch_size
        gradient_file = f"{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
        if os.path.exists(gradient_file):
            tmp_gradients = np.load(gradient_file)
            gradients.append(tmp_gradients[idx % args.batch_size])
    gradients = np.array(gradients)
    if len(gradients) == 0:
        return {}
    
    # randomly assign labels as 0 or 1
    labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
    # reverse the gradients for the 0 labels
    mask = np.copy(labels)
    mask[labels == 0] = -1
    mask = mask.reshape(-1, 1)
    gradients = gradients*mask
    train_num = int(len(gradients)*0.8)
    train_gradients, train_labels = gradients[:], labels[:]
    test_gradients, test_labels = gradients[train_num:], labels[train_num:]
    print("Train size for logistic regression", len(train_gradients))

    # train a logistic regression model
    clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4, solver='liblinear') # 
    clf.fit(train_gradients, train_labels)
    print("Linear regression score: ", clf.score(test_gradients, test_labels))
    proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
    coef = projection_matrix @ proj_coef.flatten()
    print("L2 norm", np.linalg.norm(coef))
    coef = coef*scale / np.linalg.norm(coef)
    print("L2 norm", np.linalg.norm(coef))

    new_state_dict = generate_state_dict(lm.model, state_dict, coef, device=lm.model.device)
    pretrain_state_dict = state_dict
    finetuned_state_dict = new_state_dict
    lm.model.load_state_dict(pretrain_state_dict)
    lm.model.load_state_dict(finetuned_state_dict, strict=False)

    summary = trainer.validate(lm, datamodule=data_module)[0]
    return summary

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-")
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
                                                    device_map={"": args.devices[0]}) # assume single-gpu for now
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    data_module = AlpacaDataModule(tokenizer=tokenizer,
                                data_path="./data/alpaca_data/alpaca_final.pkl",
                                dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                                task_idxes=list(range(38)),
                                batch_size = 8,
                                inference_batch_size = 8,
                                context_length=256)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    if args.load_model_dir is not None:
        load_model_dir = os.path.join("external_lightning_logs", args.load_model_dir)
        save_name = "Alpaca_{}".format(model_key) + \
                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                    (f"_{args.save_name}" if args.save_name else "") + \
                    f"_dim_{args.project_dimension}_run_{args.run}" 
        file_dir = os.path.join("./results/", save_name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        '''Deprecated'''
        # if os.path.exists(load_model_dir + ".ckpt"):
        #     gradient_dir = "./gradients/" + save_name
        #     lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
        #                             lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
        #                             intialize_project_matrix=args.project_gradients, run_seed=args.run, 
        #                             project_dim=args.project_dimension, gradients_dir=gradient_dir, use_sgd=True)
        '''Deprecated'''
        gradient_dir = "./gradients/" + save_name
        lm =  AlpacaModel(model, tokenizer, model_type, 
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                        project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True)
        # load huggingface checkpoint
        load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
        if os.path.exists(load_model_dir):
            state_dict = torch.load(load_model_dir, map_location=lm.model.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from checkpoint from ", load_model_dir)
    else:
        save_name = "Alpaca_{}".format(model_key) + \
                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                    (f"_{args.save_name}" if args.save_name else "") + \
                    f"_dim_{args.project_dimension}_run_{args.run}" + "_pretrained"
        file_dir = os.path.join("./results/", save_name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        gradient_dir = "./gradients/" + save_name 
        lm =  AlpacaModel(model, tokenizer, model_type, 
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                        project_dim=args.project_dimension, use_sgd=True)

    state_dict = {key: val.clone() for key, val in lm.model.state_dict().items()}

    pretrain_norm = compute_norm(state_dict)
    print("Norm of the original model", pretrain_norm)
    scale = pretrain_norm * args.scale

    project_matrix = lm.project_matrix 
    args.accumulate = 1; args.epochs = 0; args.enable_checkpointing = True
    default_root_dir = "external_lightning_logs/" + save_name
    if args.use_qlora:
        from lightning.pytorch.plugins import BitsandbytesPrecision
        # this will pick out the compute dtype automatically, by default `bfloat16`
        quant_precision = BitsandbytesPrecision(mode="nf4-dq")
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, # precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False, plugins=quant_precision
            )
    else:
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False
            )
    
    start_time = time.time()
    if args.load_sample_task_dir is not None:
        sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(args.load_sample_task_dir))

        count = 0
        with open(sampled_task_dir, "r") as f:
            for line in f.readlines():  
                print(line)              
                train_dataset = data_module.train_dataset
                skills = [tmp_data['skill'] for tmp_data in train_dataset.data]
                skill_list = data_module.skills
                task_num = len(skill_list)

                subset_idxes = [int(idx) for idx in line.strip().split()]
                tmp_skill_list = [skill_list[i] for i in subset_idxes]
                data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]

                tmp_start_time = time.time()
                summary = evaluate_subset(args, trainer, lm, data_module, data_idxes, state_dict, project_matrix, scale, gradient_dir)
                tmp_end_time = time.time()
                if not summary:
                    continue              

                print(f"Time for solving logistic regression and evaluation on one data set: ", tmp_end_time - tmp_start_time)  
                
                # save indexes 
                result_datapoint = {
                    "Data indices": " ".join([str(idx) for idx in subset_idxes])
                ,
                }
                for key, val in summary.items():
                    result_datapoint[key] = val
                file_name = os.path.join(file_dir, "results.csv")
                add_result_to_csv(result_datapoint, file_name)
                count += 1
                if count >= args.number_of_subsets:
                    break
    else:
        sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(save_name))
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()
        for _ in range(args.number_of_subsets):
            # Solve linear model
            train_dataset = data_module.train_dataset
            skills = [tmp_data['skill'] for tmp_data in train_dataset.data]
            skill_list = data_module.skills
            task_num = len(skill_list)

            subset_idxes = np.random.choice(task_num, int(args.subset_size*task_num), replace=False)
            subset_idxes.sort()
            tmp_skill_list = [skill_list[i] for i in subset_idxes]
            data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]
            summary = evaluate_subset(args, trainer, lm, data_module, data_idxes, state_dict, project_matrix, scale, gradient_dir)
            if not summary:
                continue

            # save indexes 
            result_datapoint = {
                "Data indices": " ".join([str(idx) for idx in subset_idxes])
            }
            for key, val in summary.items():
                result_datapoint[key] = val
            file_name = os.path.join(file_dir, "results.csv")
            add_result_to_csv(result_datapoint, file_name)

            with open(sampled_task_dir, "a") as f:
                f.write(" ".join([str(idx) for idx in subset_idxes]) + "\n")
    end_time = time.time()
    print(f"Time for solving logistic regression on {args.number_of_subsets} subsets: ", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_model_dir", type=str, default=None)

    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.02)

    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--number_of_subsets", type=int, default=100)
    parser.add_argument("--subset_size", type=float, default=0.5)
    parser.add_argument("--load_sample_task_dir", type=str, default=None)

    parser.add_argument("--use_qlora", action="store_true")
    args = parser.parse_args()
    main(args)