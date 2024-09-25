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
    devices = [0]
    accumulate = 1
    strategy = None
    precision = 32
    lr = 3e-4
    disable_checkpointing = False

    # projections
    project_dim = 200
    create_projection = True
    run = 0

    train_lora = True
    lora_rank = 16
    lora_alpha = 128

    load_model_dir = "flan_t5_base_gsm8k_ft_cot_lora_r_16_new_run_0/epoch_epoch=14"

    num_clusters = 200
    subset_size = 0.5

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

lm.completion_metadata = cm
device = torch.device(f"cuda:{args.devices[0]}")
lm.to(device)

# %%
for batch in train_loader:
    for key, value in batch.items():
        print(key, value.shape)
    break


# %%
# Compute FLOPs
from ptflops import get_model_complexity_info

def generate_batch(input_res):
    data = {}
    for key in ['input_ids', 'attention_mask']:
        dtype = torch.long
        data[key] = torch.randint(0, 2, size=(input_res[0], input_res[1]), dtype=dtype).to(device)
    for key in [ 'labels', 'decoder_attention_mask']:
        dtype = torch.long
        data[key] = torch.randint(0, 2, size=(input_res[0], input_res[2]), dtype=dtype).to(device)
    return data

macs, params = get_model_complexity_info(model, (1, 64, 256), as_strings=True, input_constructor=generate_batch,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

'''Computational complexity:       30.51 GMac'''

# %%
from torch import nn

class Projection(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(Projection, self).__init__()  
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.projection(x)
    
gradient_dim = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        gradient_dim += param.numel()

proj = Projection(gradient_dim, 400)

macs, params = get_model_complexity_info(proj, (1, gradient_dim), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

''' Computational complexity:   
Dim 50  132.71 MMac
Dim 100    265.42 MMac
Dim 200    530.85 MMac
Dim 400    1061.71 MMac
'''

# %%

# train samples: strategy_qa 12824
# test samples: 687

one_forward = 30.51
num_train_samples = 12824
num_test_samples = 687
num_models = 1000
clusters = 1
num_tasks = 100
subset_size = 0.5
num_epochs = 20

train_one_model = num_epochs * (one_forward * num_train_samples * subset_size) * 2
eval_one_model = (one_forward * num_test_samples)

total_macs = (train_one_model + eval_one_model) * num_models 
print(f"Total Macs: {total_macs:.2f} GMacs")

# %%
'''
Fast estimation:
    training base models 
    estimate gradients + project the gradients
    evaluate the performances 
'''
one_project = (265.42 / 1000)*4
num_fast_models = 9

projct_one_epoch = one_project * num_train_samples

training_base_models = train_one_model * num_fast_models
estimate_gradients = (one_forward* num_train_samples*2 + projct_one_epoch)*num_fast_models
evaluate_mtl_performances  = ((one_forward * num_test_samples) * num_models)*num_fast_models
training_cluster_models = train_one_model * clusters

total_macs = training_base_models + estimate_gradients + evaluate_mtl_performances + training_cluster_models
print(f"Total Macs: {total_macs:.2f} GMacs")


# %%
print("Naive MTL: {:.2f} GMacs".format(train_one_model))

# %% 

forward_models = 1850

total_macs = train_one_model * (forward_models)
print(f"Forward Total Macs: {total_macs:.2f} GMacs")

# %%
backward_models = 2000
total_macs = train_one_model * (backward_models)
print(f"Backward Total Macs: {total_macs:.2f} GMacs")

# %%
# TAG
tag = 15 * (59.57 * num_train_samples*num_tasks*num_tasks +  59.57 * num_train_samples*2)
total_macs = tag 
print(f"TAG Total Macs: {tag:.2f} GMacs")