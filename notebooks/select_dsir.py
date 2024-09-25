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
    dataset_key = "commonsense_qa"
    model_key = "flan_t5_base"
    train_key = "ft_cot"
    batch_size = 8
    preset_key = "ft_cot"
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
    lora_rank = 4
    lora_alpha = 32

    load_model_dir = "flan_t5_base_multiarith_ft_cot_lora_r_4/lightning_logs/version_0/checkpoints/epoch=19-step=51400"

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

data_module.setup("fit")


# %%
from typing import List, Optional, Dict, Callable, Union, Iterable
import hashlib
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk import ngrams as get_ngrams
import numpy as np

from data_selection.base import (
        DSIR,
        default_load_dataset_fn,
        default_parse_example_fn,
        _iterate_virtually_sharded_dataset,
)

from data_selection.utils import parallelize
from data_selection import HashedNgramDSIR

wpt = WordPunctTokenizer()

def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets

def get_ngram_counts(line: str,
                     n: int = 2,
                     num_buckets: int = 10000,
                     counts: Optional[np.ndarray] = None,
                     tokenizer: Callable = wpt.tokenize) -> np.ndarray:
    '''Return ngram count features given a string.

    Args:
        line: string to get ngram counts from
        n: n in ngrams
        num_buckets: number of buckets to hash ngrams into
        counts: pre-initialized counts array
        tokenizer: tokenization function to use. Defaults to word_tokenize from nltk
    '''
    words = tokenizer(line.lower())

    if counts is None:
        counts = np.zeros(num_buckets, dtype=int)

    for w in words:
        counts[hash_buckets(w, num_buckets=num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in list(get_ngrams(words, i)):
            ng = ' '.join(ng)
            counts[hash_buckets(ng, num_buckets=num_buckets)] += 1
    return counts

def get_virtually_sharded_datasets(datasets: List[str], num_proc):
    """Return virtual shard parameters."""
    num_proc_per_shard = max(1, num_proc // len(datasets))
    if num_proc >= len(datasets):
        remainder = num_proc % len(datasets)
    else:
        remainder = 0

    overall_idx = 0
    shard_params = []
    for i, dataset in enumerate(datasets):
        curr_num_proc = num_proc_per_shard
        if i < remainder:
            curr_num_proc += 1
        for j in range(curr_num_proc):
            shard_params.append({'path': dataset, 'shard_idx': j, 'num_shards': curr_num_proc, 'overall_idx': overall_idx})
            overall_idx += 1
    return shard_params

def fit_bow(paths: List[str], num_proc, num_buckets, ngrams, tokenizer,
            num_tokens_to_fit: Optional[int],
            dataset = None,
            parse_example_fn: Callable[[Dict], str]= None):

        sharded_datasets = get_virtually_sharded_datasets(paths, num_proc)

        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']

            counts = np.zeros(num_buckets).astype(int)
            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                if parse_example_fn is not None:
                    text = parse_example_fn(ex)
                else:
                    text = ex
                counts = get_ngram_counts(text,
                                          n=ngrams,
                                          num_buckets=num_buckets,
                                          counts=counts,
                                          tokenizer=tokenizer)

                if num_tokens_to_fit is not None and counts.sum() > num_tokens_to_fit // len(sharded_datasets):
                    break

            return counts

        all_counts = parallelize(job, sharded_datasets, num_proc)
        counts = sum(all_counts)

        return counts

# %%
args.n_grams = 2
args.num_buckets = 10000
args.num_proc = 8
def parse_example_fn(ex):
    ex["labels"][ex["labels"] == -100] = tokenizer.pad_token_id
    inputs = tokenizer.decode(ex["input_ids"], skip_special_tokens=True) 
    targets = tokenizer.decode(ex["labels"], skip_special_tokens=True)
    return inputs + " " + targets

raw_probs = fit_bow([f"{args.dataset_key}_{args.preset_key}"], args.num_proc, args.num_buckets, args.n_grams, wpt.tokenize, 100000 * args.num_buckets, data_module.train_dataset, parse_example_fn)
raw_probs = raw_probs / raw_probs.sum()
target_probs = fit_bow([f"{args.dataset_key}_{args.preset_key}"], args.num_proc, args.num_buckets, args.n_grams, wpt.tokenize, None, data_module.test_dataset, parse_example_fn)
target_probs = target_probs / target_probs.sum()
log_diff = np.log(target_probs + 1e-8) - np.log(raw_probs + 1e-8)

# %%
def featurizer(text):
    return get_ngram_counts(text, tokenizer=wpt.tokenize, num_buckets=args.num_buckets, n=args.n_grams)

scores = []
for ex in data_module.train_dataset:
    text = parse_example_fn(ex)
    features = featurizer(text)
    score = np.dot(log_diff, features)
    scores.append(score)
scores = np.array(scores)

# %%
np.save(f"./scores/{args.dataset_key}_{args.preset_key}_dsir_scores.npy", scores)

# %%
index_dir = "./data_indices"
indices = np.argsort(scores)
for ratio in np.arange(0.7, 1.0, 0.05):
    num_indices = int(ratio * len(indices))
    with open(f"{index_dir}/{args.dataset_key}_{args.preset_key}_dsir_ratio_{int(ratio*100)}.txt", "w") as f:
        f.write(" ".join([str(idx) for idx in indices[-num_indices:]]))