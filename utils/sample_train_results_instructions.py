import argparse
import logging
import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def main(args):
    sampled_task_dir = os.path.join("./sampled_indices", "Instruction_{}.txt".format(args.save_name))
    if not os.path.exists(sampled_task_dir):
        f = open(sampled_task_dir, "w")
        f.close()

    for _ in range(args.number_of_subsets):  
        # load clusters
        subset_idxes = np.random.choice(args.task_num, int(args.subset_size*args.task_num), replace=False)
        
        os.system('''
            python custom_train_alpaca.py --train_instruction --model_key {}\
            --lr {} --batch_size {} --max_length {} --epochs {} --precision {}\
            --train_lora --lora_rank {} --lora_alpha {}\
            --strategy fsdp --devices {} --runs 1 --task_idxes {} --save_name {} --write_results --accumulate 1
            '''.format(
            args.model_key,
            args.lr,  args.batch_size, args.max_length, args.epochs, args.precision,
            args.lora_rank, args.lora_alpha, 
            " ".join([str(idx) for idx in args.devices]),
            " ".join([str(idx) for idx in subset_idxes]),
            args.save_name
        ))

        with open(sampled_task_dir, "a") as f:
            f.write(" ".join([str(idx) for idx in subset_idxes]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--task_num", type=int, default=1729)
    parser.add_argument("--number_of_subsets", type=int, default=1000)
    parser.add_argument("--subset_size", type=float, default=0.05)
    parser.add_argument("--save_name", type=str, default="true")

    args = parser.parse_args()
    main(args)