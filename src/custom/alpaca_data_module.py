import pytorch_lightning as pl
import torch
import os
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

class AlpacaTorchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, is_eval, model_type="decoder"):
        self.data = tokenized_data
        self.is_eval = is_eval
        self.model_type = model_type

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.is_eval:
            if self.model_type == "decoder":
                
                return {
                    "input_ids": data['tokenized']['input_ids'][0],
                    "attention_mask": data['tokenized']['attention_mask'][0],
                    "skill": data['skill'],
                }
            else:
                return {
                    "input_ids": data['input_ids'][0],
                    "attention_mask": data['attention_mask'][0],
                    "decoder_attention_mask": data['decoder_attention_mask'][0],
                    "labels": data['labels'][0],
                    "skill": data['skill'],
                }
        else:
            if self.model_type == "decoder":
                return {
                    "input_ids": data['tokenized']['input_ids'][0],
                    "attention_mask": data['tokenized']['attention_mask'][0]
                }
            else:
                return {
                    "input_ids": data['input_ids'][0],
                    "attention_mask": data['attention_mask'][0],
                    "decoder_attention_mask": data['decoder_attention_mask'][0],
                    "labels": data['labels'][0],
                }
        
class StringDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator for samples with string data in addition to tensors."""
    def __init__(self, tokenizer, string_columns, mlm):
        super().__init__(tokenizer, mlm)
        self.string_columns = string_columns
                
    def __call__(self, examples):
        tensor_examples = [{k: v for k,v in ex.items() if k not in self.string_columns} for ex in examples]
        string_examples = [{k: v for k,v in ex.items() if k in self.string_columns} for ex in examples]
        batch = super().__call__(tensor_examples)
        counts = [len(s) for s in string_examples]
        if sum(counts) != 0:
            for col in self.string_columns:
                if col in string_examples[0]: # check that the string_column exists
                    batch[col] = [ex[col] for ex in string_examples]
        return batch
    
class StringDataCollatorSeq2Seq(DataCollatorForSeq2Seq):
    """Custom data collator for samples with string data in addition to tensors."""
    def __init__(self, tokenizer, string_columns):
        super().__init__(tokenizer)
        self.string_columns = string_columns
                
    def __call__(self, examples):
        tensor_examples = [{k: v for k,v in ex.items() if k not in self.string_columns} for ex in examples]
        string_examples = [{k: v for k,v in ex.items() if k in self.string_columns} for ex in examples]
        batch = super().__call__(tensor_examples)
        counts = [len(s) for s in string_examples]
        if sum(counts) != 0:
            for col in self.string_columns:
                if col in string_examples[0]: # check that the string_column exists
                    batch[col] = [ex[col] for ex in string_examples]
        return batch


class AlpacaDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokenizer,
        data_path,
        dev_split_path, 
        task_idxes,
        batch_size= 8,
        inference_batch_size=32,
        context_length=512,
        downsample=None,
        model_type="decoder"
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data_path = data_path
        self.dev_split_path = dev_split_path
        self.task_idxes = task_idxes
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.downsample = downsample
        self.model_type = model_type
        
    def _get_tokenized_val(self):
        eval_data = pd.DataFrame()
        for i, s in enumerate(self.skills):
            dev_samples = self.data.loc[self.dev_split[s]]
            eval_data = pd.concat([eval_data, dev_samples])

        if self.model_type == "decoder":
            tokenized =[{"skill": skill, 
                     "tokenized": self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.context_length, truncation=True)} 
                     for (skill, text) in eval_data[['skill', 'text']].values] 
            return AlpacaTorchDataset(tokenized, True, model_type=self.model_type) 
        else:
            assert self.model_type == "encoder_decoder"
            print("Loading encoder-decoder model")
            tokenized = []
            for (skill, instruction, input, output) in eval_data[['skill', 'instruction', 'input', 'output']].values: 
                input_text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                output_text = output

                it = self.tokenizer(
                    input_text,
                    padding="max_length",
                    max_length=self.context_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = it["input_ids"]
                attention_mask = it["attention_mask"]
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                lt = self.tokenizer(
                    output,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                label_ids = lt["input_ids"]
                decoder_attention_mask = lt["attention_mask"]
                label_ids[~decoder_attention_mask.bool()] = -100
                result.update({
                    "decoder_attention_mask": decoder_attention_mask,
                    "labels": label_ids,
                })
                result.update({"skill": skill})
                tokenized.append(result)
            return AlpacaTorchDataset(tokenized, True, model_type=self.model_type)

    def _get_tokenized_train(self):        
        all_data = pd.DataFrame()
        for i, s in enumerate(self.skills):
            s_samples = self.data.loc[(self.data.skill == s) & (~self.data.index.isin(self.dev_split[s]))]
            all_data = pd.concat([all_data, s_samples])

        if self.model_type == "decoder":
            tokenized =[{"skill": skill, 
                        "tokenized": self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.context_length, truncation=True)} 
                        for (skill, text) in all_data[['skill', 'text']].values] 
            return AlpacaTorchDataset(tokenized, False, model_type=self.model_type)
        else:
            assert self.model_type == "encoder_decoder"
            tokenized = []
            for (skill, instruction, input, output) in all_data[['skill', 'instruction', 'input', 'output']].values: 
                input_text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                output_text = output

                it = self.tokenizer(
                    input_text,
                    padding="max_length",
                    max_length=self.context_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = it["input_ids"]
                attention_mask = it["attention_mask"]
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                lt = self.tokenizer(
                    output_text,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                label_ids = lt["input_ids"]
                decoder_attention_mask = lt["attention_mask"]
                label_ids[~decoder_attention_mask.bool()] = -100
                result.update({
                    "decoder_attention_mask": decoder_attention_mask,
                    "labels": label_ids,
                })
                result.update({"skill": skill})
                tokenized.append(result)
            return AlpacaTorchDataset(tokenized, False, model_type=self.model_type)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        skills = batch["skill"] if "skill" in batch else None 
        batch = {k: v for k, v in batch.items() if k != "skill"}
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if skills is not None:
            batch["skill"] = skills
        return batch

    def setup(self, stage: str):
        with open(self.dev_split_path, "rb") as f:
            self.dev_split = pickle.load(f)
        self.data = pd.read_pickle(self.data_path)              
        self.skills = sorted(self.data.skill.unique()) # always in alphabetical order 
        self.number_of_tasks = len(self.skills)
        self.skills = [self.skills[i] for i in self.task_idxes]

        self.train_dataset = self._get_tokenized_train()
        if self.downsample is not None:
            rng = np.random.default_rng(1024)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, rng.choice(len(self.train_dataset), self.downsample, replace=False))
        self.test_dataset = self._get_tokenized_val()
    
    def train_dataloader(self):
        string_columns = ["skill"]
        if self.model_type == "decoder":
            data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)
        else:
            data_collator = StringDataCollatorSeq2Seq(self.tokenizer, string_columns)
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
        string_columns = ["skill"]
        if self.model_type == "decoder":
            data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)
        else:
            data_collator = StringDataCollatorSeq2Seq(self.tokenizer, string_columns)
        sampler = SequentialSampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )

    def test_dataloader(self):
        string_columns = ["skill"]
        if self.model_type == "decoder":
            data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)
        else:
            data_collator = StringDataCollatorSeq2Seq(self.tokenizer, string_columns)
        sampler = SequentialSampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )