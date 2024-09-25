# %%
''' Test GLUE data '''
from datasets import load_metric

metric = load_metric("super_glue", "copa")


# %%
from src.custom.glue_task_constants import task_to_benchmark, task_to_instruction_template, task_is_generative_task
from src.utils.template_utils import apply_template
from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

task_name = "multirc"
do_eval = True
do_predict = True
benchmark_name = task_to_benchmark[task_name] # Test set does not have labels

if benchmark_name is not None:
    print(f"loading dataset {benchmark_name}/{task_name}....")
    raw_datasets = load_dataset(benchmark_name, task_name)
    dataset_templates = DatasetTemplates(benchmark_name, task_name)
    print(f"{benchmark_name}/{task_name} loading completed!")
else:
    print(f"loading dataset {task_name}....")
    raw_datasets = load_dataset(task_name)
    dataset_templates = DatasetTemplates(task_name)
    print(f"{task_name} loading completed!")

keys = list(dataset_templates.name_to_id_mapping.keys())
if task_is_generative_task[task_name]:
    templates = [dataset_templates[key] for key in keys if 'ROUGE' in dataset_templates[key].metadata.metrics]
else:
    templates = [dataset_templates[key] for key in keys]

i = 0
template = templates[i]
while not template.answer_choices and (i + 1) < len(templates):
    i += 1
    template = templates[i]

# unifying labels space
template = Template(name="defined", jinja=task_to_instruction_template[task_name], reference="", answer_choices = template.answer_choices)

if "train" not in raw_datasets:
    raise ValueError("--do_train requires a train dataset")
train_dataset = raw_datasets["train"]

valid_name = "validation_matched" if task_name == "mnli" else "validation"
test_name = "test_matched" if task_name == "mnli" else "test"
if do_eval:
    if valid_name not in raw_datasets:
        print("No validation dataset, using test set")
        if test_name not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[test_name]
    else:
        eval_dataset = raw_datasets[valid_name] 
else:
    eval_dataset = []

if do_predict:
    if test_name not in raw_datasets:
        print("No test dataset, using validation set")
        if valid_name not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[valid_name]
    else:
        predict_dataset = raw_datasets[test_name] 
else:
    predict_dataset = []

print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))

# Prepare validation and test dataset
column_names = train_dataset.column_names
if "input" in column_names:
    column_names.remove("input")
    train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
    train_dataset.remove_columns(["old_input"])
else:
    train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

if do_eval:
    column_names = eval_dataset.column_names
    if "input" in column_names:
        column_names.remove("input")
        eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
        eval_dataset.remove_columns(["old_input"])
    else:
        eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

if do_predict:
    column_names = predict_dataset.column_names
    if "input" in column_names:
        column_names.remove("input")
        predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
        predict_dataset.remove_columns(["old_input"])
    else:
        predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

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
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from src.custom.model import Model

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class args:
    dataset_key = "svamp"
    model_key = "EleutherAI/gpt-neo-125M"
    train_key = "ft_cot"
    batch_size = 8
    preset_key = "ft_cot_t70_64aug"
    inference_batch_size = None
    devices = [0, 1]
    accumulate = 1
    strategy = None
    precision = 32
    lr = 3e-4
    disable_checkpointing = False


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
elif "gpt" in model_key:
    hf_key = model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
else:
    raise NotImplementedError(model_key)

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
                            inference_batch_size=inference_batch_size, num_workers=8, append_eos=append_eos,
                            data_index_dir="svamp_test")

# %%
from src.custom.glue_data_module import GLUEDataModule

data_module = GLUEDataModule(
        task_name="cola",
        tokenizer=tokenizer,
        batch_size=8,
        inference_batch_size=None,
        max_input_length=256)
data_module.setup(stage="fit")

# %%
train_dataloader = data_module.train_dataloader()
for batch in train_dataloader:
    print(batch)
    break

# %%
from src.custom.toxigen_data_module import ToxiGenDataModule

data_module = ToxiGenDataModule(tokenizer=tokenizer,
                                      data_path="./data/eval/toxigen",
                                      batch_size=8,
                                      inference_batch_size=None,
                                      context_length=256,
                                      dev_split_ratio=0.1,
                                      load_full_as_train=True)
data_module.setup(stage="fit")


# %%
from src.custom.truthfulqa_data_module import TruthfulQADataModule

data_module = TruthfulQADataModule(tokenizer=tokenizer,
                                      data_path="./data/eval/truthfulqa",
                                      batch_size=8,
                                      inference_batch_size=None,
                                      context_length=256,
                                      dev_split_ratio=0.1,
                                      load_full_as_train=True,
                                      use_preset=True)
data_module.setup(stage="fit")


# %%
from src.custom.alpaca_data_module import AlpacaDataModule

data_module = AlpacaDataModule(tokenizer=tokenizer,
                               data_path="./data/alpaca_data/alpaca_final.pkl",
                               dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                               task_idxes=list(range(38)),
                               batch_size = 8,
                               inference_batch_size = 8,
                               context_length=256)
data_module.setup(stage="fit")

# %%
train_dataloader = data_module.train_dataloader()

for batch in train_dataloader:
    print((batch["input_ids"] == tokenizer.pad_token_id).sum(dim=1))

# %%
import pandas as pd
alpaca_data = pd.read_pickle("./data/alpaca_data/alpaca_final.pkl")


# %%
from src.custom.instruction_data_module import InstructionDataModule

# 1729
data_module = InstructionDataModule(tokenizer=tokenizer,
                                    task_idxes=list(range(1729)),
                                    batch_size=8,
                                    inference_batch_size=8,
                                    context_length=256)
data_module.setup(stage="fit")

# %%
for i, s in enumerate(data_module.skills):
    s_samples = data_module.data.loc[(data_module.data.skill == s) & (~data_module.data.index.isin(data_module.dev_split[s]))]
    print(len(s_samples))

# %%
train_loader = data_module.train_dataloader()
# %%
for batch in train_loader:
    print(batch)
    break

# %%
max_len = 0
valid_loader = data_module.val_dataloader()
for batch in valid_loader:
    max_len += ((batch.labels!=-100).sum(1) == 256).sum().item()
    print((batch.labels!=-100).sum(1))
    print(batch['skill'])


# %%
import torch.nn.functional as F
device = torch.device('cuda')

input_ids = batch['input_ids'].to(device)
labels = batch["labels"].to(device)
model.to(device)

outputs = model(input_ids, labels=labels)
lm_logits = outputs.logits 
shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

# %%
context_length = 255

losses = loss.view(-1, context_length)
keep = losses != 0
losses = (losses).sum(dim = 1) / keep.sum(dim = 1)

# %%
import numpy as np
import pandas as pd

true_results_df = pd.read_csv("./results/Alpaca_EleutherAI-gpt-neo-125M_lora_r_4_true/results.csv", index_col=0)
estimated_results_df = pd.read_csv("./results/Alpaca_EleutherAI-gpt-neo-125M_lora_r_4_dim_100_run_0/results.csv", index_col=0)
# %%
estimated_results = []
true_results = []
for i in range(38):
    true_results_df = true_results_df[true_results_df["Target task"]==i]
    for data_idxes in true_results_df["Data indices"]:
        estimated_results.append(estimated_results_df[estimated_results_df["Data indices"]==data_idxes][estimated_results_df.columns[i+1]].values[0])
    true_results.append(true_results_df["Loss"].values)

estimated_results = np.array(estimated_results)
true_results = np.concatenate(true_results)
np.square(np.abs(estimated_results - true_results)/np.abs(true_results)).mean()