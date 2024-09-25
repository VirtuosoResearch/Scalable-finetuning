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
from src.custom.instruction_data_module import InstructionDataModule
from peft import get_peft_model, LoraConfig

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

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

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-")

    if "gpt" in args.model_key or "Llama" in args.model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(args.model_key)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    
    # load huggingface checkpoint
    load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
    if os.path.exists(load_model_dir):
        state_dict = torch.load(load_model_dir, map_location=model.device)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded model from checkpoint from ", load_model_dir)

    if args.train_instruction:
        task_idxes = list(range(1729))
        data_module = InstructionDataModule(tokenizer=tokenizer,
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length)
    else:
        # Only load alpaca dataset
        task_idxes = list(range(38))
        data_module = AlpacaDataModule(tokenizer=tokenizer,
                            data_path="./data/alpaca_data/alpaca_final.pkl",
                            dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length)
    data_module.setup(stage="fit")

    save_name = ("Instruction_{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                (f"_task_{args.target_task}") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") +\
                (f"_sample_subsets_{args.subset_size}_{args.number_of_subsets}") + f"_{args.save_name}"
    
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    state_dict = {key: val.clone() for key, val in model.state_dict().items()}
    pretrain_norm = compute_norm(state_dict)
    print("Norm of the original model", pretrain_norm)

    gradient_dim = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_dim += param.numel()

    np.random.seed(args.run)
    project_dim = args.project_dimension
    project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
    project_matrix *= 1 / np.sqrt(project_dim)
    
    sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(save_name))
    if not os.path.exists(sampled_task_dir):
        f = open(sampled_task_dir, "w")
        f.close()

    for _ in range(args.number_of_subsets):
        # sample task subsets
        train_dataset = data_module.train_dataset
        skills = [tmp_data['skill'] for tmp_data in train_dataset.data]
        skill_list = data_module.skills
        gradient_dir = "./gradients/" + f"{args.gradient_dir}"
        task_num = len(skill_list)
        
        subset_idxes = np.random.choice(task_num, int(args.subset_size*task_num), replace=False)
        subset_idxes.sort()
        tmp_skill_list = [skill_list[i] for i in subset_idxes]
        data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]

        # collect gradients
        gradients = []
        for idx in data_idxes:
            gradient_file_idx = idx // args.batch_size
            gradient_file = f"{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
            if not os.path.exists(gradient_file): continue
            tmp_gradients = np.load(gradient_file)
            gradients.append(tmp_gradients[idx % args.batch_size])
        gradients = np.array(gradients)

        # Solve linear model
        #   randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
        #   reverse the gradients for the 0 labels
        mask = np.copy(labels)
        mask[labels == 0] = -1
        mask = mask.reshape(-1, 1)
        gradients = gradients*mask
        train_num = int(len(gradients)*0.8)
        train_gradients, train_labels = gradients[:train_num], labels[:train_num]
        test_gradients, test_labels = gradients[train_num:], labels[train_num:]

        clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4, solver='liblinear') 
        clf.fit(train_gradients, train_labels)
        print(clf.score(test_gradients, test_labels))

        proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
        coef = project_matrix @ proj_coef.flatten()
        print("L2 norm", np.linalg.norm(coef))

        cur_coef = (args.scale *  pretrain_norm) * coef / np.linalg.norm(coef) 
        print("Current norm of the coef", np.linalg.norm(cur_coef))
        new_state_dict = generate_state_dict(model, state_dict, cur_coef)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict

        model.load_state_dict(pretrain_state_dict)
        model.load_state_dict(finetuned_state_dict, strict=False)

        # export the large model to a temporary directory
        # model.save_pretrained(f"./exported_model/{save_name}")
        def get_state_dict(model):
            state_dict = model.state_dict()
            returned_state_dict = {}
            for k in state_dict.keys():
                if "lora" in k: 
                    returned_state_dict[k] = state_dict[k].cpu().clone()
            return returned_state_dict

        torch.save(get_state_dict(model), f"./exported_model/{save_name}.pt")

        # evaluate the model
        if args.target_task == "truthfulqa":
            if os.path.exists(f"./results/trutufulqa/{save_name}/metrics.json"):
                os.remove(f"./results/trutufulqa/{save_name}/metrics.json")
            os.system(
                """
                CUDA_VISIBLE_DEVICES={} python -m eval.truthfulqa.run_eval \
                    --data_dir data/eval/truthfulqa \
                    --save_dir results/trutufulqa/{} \
                    --model_name_or_path {}\
                    --tokenizer_name_or_path {} \
                    --adapter_path {} \
                    --lora_rank {} --lora_alpha {} \
                    --metrics mc \
                    --preset qa \
                    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
                    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
                    --eval_batch_size 20 \
                    --load_in_8bit
                """.format(
                        ",".join([str(device) for device in args.devices]),
                        save_name, args.model_key, args.model_key,
                        f"./exported_model/{save_name}",
                        args.lora_rank, args.lora_alpha)
            )
            import time
            while not os.path.exists(f"./results/trutufulqa/{save_name}/metrics.json"):
                time.sleep(30)
            with open(f"./results/trutufulqa/{save_name}/metrics.json") as f:
                summary = json.load(f)
        elif args.target_task == "mmlu":
            os.system("""
                CUDA_VISIBLE_DEVICES={} python -m eval.mmlu.run_eval \
                    --ntrain 0 \
                    --data_dir data/eval/mmlu \
                    --save_dir results/mmlu/{} \
                    --model_name_or_path {}\
                    --tokenizer_name_or_path {} \
                    --adapter_path {} \
                    --eval_batch_size 4 \
                    --load_in_8bit
                """.format(
                        ",".join([str(device) for device in args.devices]),
                        save_name, args.model_key, args.model_key,
                        f"./exported_model/{save_name}")
            )
            with open(f"./results/mmlu/{save_name}/metrics.json") as f:
                summary = json.load(f)
                cat_accs = summary["cat_acc"]
                cat_accs = {key + "_acc": val for key, val in cat_accs.items()}
                sub_accs = summary["subcat_acc"]
                sub_accs = {key + "_acc": val for key, val in sub_accs.items()}
                summary = {"average_acc": summary["average_acc"]} 
                summary.update(cat_accs)
                summary.update(sub_accs)
        elif args.target_task == "bbh":
            os.system("""
                CUDA_VISIBLE_DEVICES={} python -m eval.bbh.run_eval \
                    --data_dir data/eval/bbh \
                    --save_dir results/bbh/{} \
                    --model {}\
                    --tokenizer {} \
                    --adapter_path ./exported_model/{} \
                    --max_num_examples_per_task 40 \
                    --no_cot  
                """.format(
                    ",".join([str(device) for device in args.devices]),
                    save_name, args.model_key, args.model_key,
                    f"./exported_model/{save_name}")
            )
            with open(f"./results/trutufulqa/{save_name}/metrics.json") as f:
                summary = json.load(f)

        # save indexes 
        result_datapoint = {
            "Task indices": " ".join([str(idx) for idx in subset_idxes])
        }
        for key, val in summary.items():
            result_datapoint[key] = val
        file_name = os.path.join(file_dir, "results.csv")
        add_result_to_csv(result_datapoint, file_name)

        with open(sampled_task_dir, "a") as f:
            f.write(" ".join([str(idx) for idx in subset_idxes]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_task", type=str, default="truthfulqa")
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--load_model_dir", type=str, default="test")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_instruction", action="store_true")

    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--project_dimension", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.5)

    parser.add_argument("--gradient_dir", type=str, default="")
    parser.add_argument("--save_name", type=str, default="test")
    parser.add_argument("--number_of_subsets", type=int, default=100000)
    parser.add_argument("--subset_size", type=float, default=0.5)
    parser.add_argument("--load_sample_task_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)