# %%
import argparse
import logging
import os

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from src.custom.model import Model

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class args:
    dataset_key = "strategy_qa"
    model_key = "flan_t5_base"
    train_key = "ft_cot"
    batch_size = 8
    preset_key = "ft_cot_t70_64aug"
    inference_batch_size = None
    devices = [2]
    accumulate = 1
    strategy = None
    precision = 32
    lr = 3e-4
    disable_checkpointing = False

    # projections
    project_dim = 100
    create_projection = True
    run = 0

    train_lora = True
    lora_rank = 4
    lora_alpha = 32

    load_model_dir = None # "flan_t5_base_multiarith_ft_cot_t70_64aug_lora_r_4_run_0/epoch_epoch=19"

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
elif "gpt2" in model_key:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    hf_key = model_key.replace("_", "-")
    tokenizer = GPT2Tokenizer.from_pretrained(hf_key)
    model = GPT2LMHeadModel.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
else:
    raise NotImplementedError(model_key)

if args.train_lora:
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "k", "v"],
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=[],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

# %%
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

# %%
from torch.utils.data import DataLoader

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

# %%
cm = CompletionMetadata(model_key, completion_key, dataset_key, prediction_template=data_module.prediction_template)
lm = Model(model, tokenizer, model_type, completion_metadata=cm, truncate_early=False)
load_model_dir = args.load_model_dir

if load_model_dir is not None:
    load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
    lm = lm.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type)

device = torch.device(f"cuda:{args.devices[0]}")
lm.to(device)

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression

def generate_state_dict(model, state_dict, coef, device, removing_keys = ["shared", "lm_head"]):
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
project_dim = args.project_dim
project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
project_matrix *= 1 / np.sqrt(project_dim)

# %%
for run in range(10):
    dataset_len = len(data_module.train_dataset)
    subset_idxes = np.random.choice(dataset_len, int(0.5*dataset_len), replace=False)
    subset_idxes.sort()

    gradient_dir = f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/run_{args.run}" if args.load_model_dir is not None else \
        f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}_pretrained/run_{args.run}"
    gradients = []
    for idx in subset_idxes:
        gradient_file_idx = idx // args.batch_size
        gradient_file = f"{gradient_dir}/train_batch_{gradient_file_idx}_gradients.npy"
        if not os.path.exists(gradient_file):
            continue
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

    clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4) # solver='liblinear' 
    clf.fit(train_gradients, train_labels)
    print(clf.score(test_gradients, test_labels))

    proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
    coef = project_matrix @ proj_coef.flatten()
    print("L2 norm", np.linalg.norm(coef))

    # def eval_output(model, task_idx, train_loader, device, pretrain_state_dict, finetuned_state_dict, steps = 200):
    for scale in np.arange(0.2, 0.0, -0.02):
        cur_coef = (scale *  pretrain_norm) * coef / np.linalg.norm(coef) 
        print("Current norm of the coef", np.linalg.norm(cur_coef))
        new_state_dict = generate_state_dict(lm.model, state_dict, cur_coef, device)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict
        removing_keys = ["shared", "lm_head"]
        train_loader = DataLoader(
                    data_module.train_dataset.select(subset_idxes),
                    batch_size=data_module.batch_size,
                    num_workers=data_module.num_workers,
                    shuffle=False)

        '''
        Compute gradients: for a single task idx
        '''
        model.eval()

        parameters = [param for param in model.parameters() if param.requires_grad]

        # For decoupling trainer, the train loader only loads propagated features and labels
        diffs = 0; counts = 0; sum=0
        for batch in train_loader:
            model.load_state_dict(pretrain_state_dict)
            batch = {k: v.to(lm.device) for k, v in batch.items()}
            # output = lm.training_step(batch, 0)
            
            kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            if lm.model_type == "encoder_decoder":
                kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
            loss1 = lm.model(**kwargs)["loss"].detach().item(); print(loss1)
            outputs = lm.model(**kwargs)["logits"]

            # get the gradient of the output
            labels = kwargs["labels"]

            pretrain_outputs = torch.zeros_like(labels, dtype=torch.float32)
            pretrain_gradients = torch.zeros_like(labels, dtype=torch.float32)
            for i in range(len(labels)):
                tmp_mask = labels[i] != -100
                tmp_logits = outputs[i][tmp_mask]
                tmp_probs = torch.softmax(tmp_logits, dim=-1)
                tmp_labels = labels[i][tmp_mask]

                tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
                tmp_outputs = tmp_outputs - 1e-3
                tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
                for j in range(len(tmp_outputs)):
                    tmp_gradients = torch.autograd.grad(tmp_outputs[j], parameters, retain_graph=True, create_graph=False)
                    tmp_gradients = dict(
                        (name, grad) for name, grad in zip([param[0] for param in model.named_parameters() if param[1].requires_grad], tmp_gradients)
                    )
                    dot_product = 0
                    for key, param in model.named_parameters():
                        if param.requires_grad and (not any([rkey in key for rkey in removing_keys])):
                            dot_product += (tmp_gradients[key]*(finetuned_state_dict[key]-pretrain_state_dict[key])).sum().cpu().item()
                    pretrain_gradients[i][j] = (dot_product)
                    if dot_product > 10:
                        print(i, j)
                    pretrain_outputs[i][j] = tmp_outputs[j]
            outputs.detach(); 
            pretrain_outputs = pretrain_outputs.detach()

            model.load_state_dict(pretrain_state_dict)
            model.load_state_dict(finetuned_state_dict, strict=False)
            loss2 = lm.model(**kwargs)["loss"].detach().item(); print(loss2)
            sum += loss2-loss1
            outputs = lm.model(**kwargs)["logits"].detach()
            finetuned_outputs = torch.zeros_like(labels, dtype=torch.float32)
            for i in range(len(labels)):
                tmp_logits = outputs[i]
                tmp_probs = torch.softmax(tmp_logits, dim=-1)
                tmp_labels = labels[i]

                tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
                tmp_outputs = tmp_outputs - 1e-3
                tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
                finetuned_outputs[i] = tmp_outputs

            mask = torch.logical_and(finetuned_outputs != 0, pretrain_outputs != 0)
            pretrain_outputs = pretrain_outputs[mask]
            pretrain_gradients = pretrain_gradients[mask]
            finetuned_outputs = finetuned_outputs[mask]
            diff = (pretrain_outputs+pretrain_gradients-finetuned_outputs).abs() / torch.maximum(finetuned_outputs.abs(), pretrain_outputs.abs())
            diff = diff[~torch.isnan(diff)]
            print(diff)
            diffs += diff.square().mean().cpu().item()
            print(diffs)
            counts += 1
            if counts >= 10:
                break

        print(f"Average relative error", diffs/counts)