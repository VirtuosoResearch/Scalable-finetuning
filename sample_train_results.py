import argparse
import logging
import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from src.custom.model import Model
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import json
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def main(args):
    index_dir = "./data_indices"
    sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(args.load_sample_task_dir))
    count=0
    with open(sampled_task_dir, "r") as f:
        for line in f.readlines():
            # load clusters
            subset_idxes = [int(idx) for idx in line.strip().split()]
            data_idxes = []
            for idx in subset_idxes:
                tmp_idxes = np.load(f"./gradients/strategy_qa_flan_t5_base_ft_cot_t70_64aug_100/clusters_{args.num_clusters}/cluster_{idx}.npy") 
                data_idxes.append(tmp_idxes)
            data_idxes = np.concatenate(data_idxes)
            data_idxes.sort()
            subset_idxes.sort()

            tmp_data_idx_dir = f"{args.dataset_key}_{args.preset_key}_clusters_{args.num_clusters}_tmp"
            with open(f"{index_dir}/{tmp_data_idx_dir}.txt", "w") as f:
                f.write(" ".join([str(idx) for idx in data_idxes]))

            os.system('''
                python custom_train.py --dataset_key {} --model_key {} --train_key {} \
                --preset_key {} --devices {} --batch_size {} --inference_batch_size {} --epochs {} \
                {}\
                --data_index_dir {}\
                --save_name {} --runs 1 --write_results --subset_idxes {} {}
                '''.format(
                args.dataset_key, args.model_key, args.train_key, args.preset_key,
                " ".join([str(device) for device in args.devices]),
                args.batch_size, args.inference_batch_size, args.epochs,
                "--train_lora --lora_rank {} --lora_alpha {}".format(args.lora_rank, args.lora_alpha) if args.train_lora else "",
                tmp_data_idx_dir,
                args.save_name,
                " ".join([str(idx) for idx in subset_idxes]),
                "--use_qlora --optimizer 'paged_adamw_32bit' --precision 'bf16-true'" if args.use_qlora else ""
            ))
            count+=1
            if count==args.num_subsets:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="multiarith")
    parser.add_argument("--model_key", type=str, default="flan_t5_base")
    parser.add_argument("--train_key", type=str, default="ft_cot")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preset_key", type=str, default="ft_cot_t70_64aug")
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--load_sample_task_dir", type=str, default=None)
    parser.add_argument("--project_dim", type=int, default=200)
    parser.add_argument("--load_clusters", action="store_true")
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.4)
    parser.add_argument("--save_name", type=str, default="true")

    parser.add_argument("--num_subsets", type=int, default=20)
    args = parser.parse_args()
    main(args)