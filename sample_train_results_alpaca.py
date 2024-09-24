import argparse
import logging
import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def main(args):
    index_dir = "./data_indices"
    sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(args.load_sample_task_dir))
    with open(sampled_task_dir, "r") as f:
        for line in f.readlines():
            # load clusters
            subset_idxes = [int(idx) for idx in line.strip().split()]
            
            os.system('''
                python custom_train_alpaca.py --model_key {} \
                    --lr {} --batch_size {} --max_length {} --epochs {} --precision {}\
                    --train_lora --lora_rank {} --lora_alpha {}\
                    --devices {} --runs 1 --task_idxes {} --save_name {} --write_results
                '''.format(
                args.model_key,
                args.lr,  args.batch_size, args.max_length, args.epochs, args.precision,
                args.lora_rank, args.lora_alpha, 
                " ".join([str(idx) for idx in args.devices]),
                " ".join([str(idx) for idx in subset_idxes]),
                args.save_name
            ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--load_model_dir", type=str, default="flan_t5_base_multiarith_ft_cot_lora_r_4/lightning_logs/version_0/checkpoints/epoch=19-step=51400")
    parser.add_argument("--load_sample_task_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="true")

    args = parser.parse_args()
    main(args)