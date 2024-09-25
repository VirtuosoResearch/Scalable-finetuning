import pytorch_lightning as pl
import torch
import os
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
import glob
import tqdm
import random

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        
        return {
            "input_ids": data['tokenized']['input_ids'][0],
            "attention_mask": data['tokenized']['attention_mask'][0]
        }


class ToxiGenDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer,
        data_path,
        batch_size= 8,
        inference_batch_size=32,
        context_length=512,
        dev_split_ratio=0.1, 
        load_full_as_train=True,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data_path = data_path
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.dev_split_ratio = dev_split_ratio
        self.load_full_as_train = load_full_as_train
    
    def _get_tokenized_dataset(self, examples):
        tokenized =[{"tokenized": self.tokenizer(
                     examples[idx]['text'],
                     return_tensors="pt", padding="max_length", max_length=self.context_length, truncation=True)} 
                     for idx in range(len(examples))] 
        return TorchDataset(tokenized)

    def setup(self, stage=None):
        # Load the testing data
        examples = []
        prompt_files = glob.glob(os.path.join(self.data_path, "*.txt"))
        for task_file in tqdm.tqdm(prompt_files, desc="Loading prompts"):
            with open(task_file, "r") as f:
                group_name = os.path.basename(task_file).split(".")[0]
                label = group_name.split("_")[0]
                minority_group = "_".join(group_name.split("_")[1:])
                group_prompts = [line.strip() for line in f]
                random.shuffle(group_prompts)
                for prompt in group_prompts:
                    # minor cleaning: replace \\ with \
                    prompt = prompt.replace("\\\\", "\\")
                    prompt = prompt.replace("\\n", "\n")
                    examples.append({
                        "text": prompt,
                        "label": label,
                        "target_groups": [minority_group],
                    })
        
        rng = np.random.default_rng(42)
        permutation = rng.permutation(len(examples))
        train_idx, dev_idx = permutation[:int(len(examples) * (1-self.dev_split_ratio))], permutation[int(len(examples) * (1-self.dev_split_ratio)):]
        if self.load_full_as_train:
            self.train_data = examples
            self.dev_data = [examples[i] for i in dev_idx]
        else:
            self.train_data = [examples[i] for i in train_idx ]
            self.dev_data = [examples[i] for i in dev_idx]
        
        self.train_dataset = self._get_tokenized_dataset(self.train_data)
        self.dev_dataset = self._get_tokenized_dataset(self.dev_data)
    
    def train_dataloader(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        train_sampler = SequentialSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        sampler = SequentialSampler(self.dev_dataset)
        return DataLoader(
            self.dev_dataset,
            batch_size=self.inference_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )

    def test_dataloader(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        sampler = SequentialSampler(self.dev_dataset)
        return DataLoader(
            self.dev_dataset,
            batch_size=self.inference_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )
        