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


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    if args.precision == 16:
        args.precision = "bf16"
        print("Setting precision to bf16")

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

    if args.train_instruction:
        task_idxes = list(range(30))
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
                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") 
    gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "")

    if args.load_model_dir is not None:
        load_model_dir = os.path.join("external_lightning_logs", args.load_model_dir)
        if os.path.exists(load_model_dir + ".ckpt"):
            lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type,
                                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                                    intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                                    project_dim=args.project_dimension, gradient_dir=gradient_dir)
            print("Loaded model from checkpoint")
    else:
        lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                        project_dim=args.project_dimension, gradient_dir=gradient_dir)

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe"]):
        # reshape coef
        new_state_dict = {}; cur_len = 0
        for key, param in model.named_parameters():
            if not param.requires_grad: continue
            param_len = param.numel()
            if any([rkey in key for rkey in removing_keys]):
                new_state_dict[key] = state_dict[key].clone()
            else:
                new_state_dict[key] = state_dict[key].clone() + \
                    torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
        return new_state_dict

    def compute_norm(state_dict):
        norm = 0
        for key, val in state_dict.items():
            if "lora" in key:
                norm += val.clone().square().sum().item()
        return np.math.sqrt(norm)

    state_dict = {key: val.clone() for key, val in lm.model.state_dict().items()}
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
    inv_project_matrix = np.linalg.pinv(project_matrix)

    args.accumulate = 1; args.epochs = 0; args.enable_checkpointing = True
    default_root_dir = "external_lightning_logs/" + save_name + f"_eval_output_approx"
    
    if args.compute_pretrained_outputs:
        model.load_state_dict(state_dict)
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                    default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                    accumulate_grad_batches=args.accumulate, precision=args.precision,
                    enable_checkpointing=args.enable_checkpointing, inference_mode=False
        )
        lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                        project_dim=args.project_dimension, gradient_dir=gradient_dir,
                        predict_steps=args.number_of_batches)
        pretrain_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        pretrain_outputs = np.concatenate(pretrain_outputs, axis=0)

        print(pretrain_outputs.shape)
        np.save(f"./gradients/{gradient_dir}/pretrain_outputs.npy", pretrain_outputs)

    else:
        # Solve linear model
        train_dataset = data_module.train_dataset
        skills = [tmp_data['skill'] for tmp_data in train_dataset.data]
        skill_list = data_module.skills
        task_num = len(skill_list)
        np.random.seed(args.seed)
        subset_idxes = np.random.choice(task_num, int(0.75*task_num), replace=False)
        subset_idxes.sort()
        tmp_skill_list = [skill_list[i] for i in subset_idxes]
        data_idxes = [i for i in range(len(skills)) if skills[i] in tmp_skill_list]

        gradients = []
        for idx in data_idxes:
            gradient_file_idx = idx // args.batch_size
            gradient_file = f"./gradients/{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
            if not os.path.exists(gradient_file): continue
            tmp_gradients = np.load(gradient_file)
            gradients.append(tmp_gradients[idx % args.batch_size])
        gradients = np.array(gradients)
        # randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
        # reverse the gradients for the 0 labels
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
        new_state_dict = generate_state_dict(lm.model, state_dict, cur_coef)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict

        pretrain_outputs = np.load(f"./gradients/{gradient_dir}/pretrain_outputs.npy")

        data_gradients = []
        for gradient_idx, file in enumerate(os.listdir(f"./gradients/{gradient_dir}")):
            data_gradients.append(np.load(os.path.join(f"./gradients/{gradient_dir}", file)))
            if gradient_idx > 10: break
        data_gradients = np.concatenate(data_gradients, axis=0)
        data_gradients = data_gradients @ inv_project_matrix
        
        finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
        finetuned_vector = np.concatenate([vec.flatten().numpy() for vec in finetuned_vector]).reshape(1,-1)
        print("Pretrained outputs:", pretrain_outputs[:4])
        dot_product = (data_gradients * finetuned_vector).sum(axis=1)
        print("First-order term", dot_product)

        model.load_state_dict(pretrain_state_dict)
        model.load_state_dict(finetuned_state_dict, strict=False)
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                    default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                    accumulate_grad_batches=args.accumulate, precision=args.precision,
                    enable_checkpointing=args.enable_checkpointing, inference_mode=False
        )   
        lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                project_dim=args.project_dimension, gradient_dir=save_name + f"_eval_output_approx",
                predict_steps=10)
        finetuned_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
        finetuned_outputs = np.concatenate(finetuned_outputs, axis=0)
        print("Fine-tuned outputs:", finetuned_outputs[:4])

        pretrain_outputs = pretrain_outputs[:dot_product.shape[0]]
        finetuned_outputs = finetuned_outputs[:dot_product.shape[0]]

        mask = np.logical_and(pretrain_outputs != 0, finetuned_outputs != 0)
        mask = np.logical_and(mask, ~np.isnan(pretrain_outputs))
        mask = np.logical_and(mask, ~np.isnan(finetuned_outputs))
        pretrain_outputs[~mask] = 0 
        finetuned_outputs[~mask] = 0
        pretrain_outputs = pretrain_outputs.sum(axis=1)/mask.sum(axis=1)
        finetuned_outputs = finetuned_outputs.sum(axis=1)/mask.sum(axis=1)

        diff = np.abs(pretrain_outputs + dot_product - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        print("Differences:", diff)
        diffs = np.square(diff).mean()
        print("Mean Difference:", diffs)

        diff = np.abs(pretrain_outputs - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        print("Differences v2:", diff)
        diffs = np.square(diff).mean()
        print("Mean Difference v2:", diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--strategy", type=str, default=None)
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
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_instruction", action="store_true")
    parser.add_argument("--compute_pretrained_outputs", action="store_true")
    parser.add_argument("--number_of_batches", type=int, default=200)
    args = parser.parse_args()
    main(args)