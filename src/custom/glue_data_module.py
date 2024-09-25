import pytorch_lightning as pl
import torch
import os
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *

import glob
import tqdm
import random

from custom.glue_task_constants import task_to_benchmark, task_to_instruction_template, task_is_generative_task
from utils.template_utils import apply_template
from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

@dataclass
class Seq2SeqInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch
        # prepare input sources
        sources = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        labels = [instance["output"] for instance in converted_batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        return model_inputs

@dataclass
class CasualLMInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch

        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            source = source.replace("\n", " ")
            source = " ".join(source.split())
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []; label_lengths = []
        for instance in converted_batch:
            label = instance["output"]
            label = label.replace("\n", " ")
            label = " ".join(label.split())
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))
            label_lengths.append(min(len(tokenized_label), self.max_target_length))

        inputs = [source + " " + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id            
        return model_inputs

class GLUEDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_name,
        tokenizer,
        batch_size= 8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=4, # deprecated
    ):
        super().__init__()

        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size

    def setup(self, stage=None):
        task_name = self.task_name
        do_eval = True
        do_predict = True
        benchmark_name = task_to_benchmark[task_name] # Test set does not have labels

        # load task and original template
        if benchmark_name is not None:
            print(f"loading dataset {benchmark_name}/{task_name}....")
            raw_datasets = load_dataset(benchmark_name, task_name, trust_remote_code=True)
            dataset_templates = DatasetTemplates(benchmark_name, task_name)
            print(f"{benchmark_name}/{task_name} loading completed!")
        else:
            print(f"loading dataset {task_name}....")
            raw_datasets = load_dataset(task_name, trust_remote_code=True)
            dataset_templates = DatasetTemplates(task_name)
            print(f"{task_name} loading completed!")

        keys = list(dataset_templates.name_to_id_mapping.keys())
        if task_is_generative_task[task_name]:
            templates = [dataset_templates[key] for key in keys if 'ROUGE' in dataset_templates[key].metadata.metrics]
        else:
            templates = [dataset_templates[key] for key in keys]

        # load the first template with answer choices
        i = 0
        template = templates[i]
        while not template.answer_choices and (i + 1) < len(templates):
            i += 1
            template = templates[i]

        # unifying labels space
        template = Template(name="defined", jinja=task_to_instruction_template[task_name], reference="", answer_choices = template.answer_choices)
        if self.task_name == "copa":
            template.answer_choices = "First ||| Second"
        self.template = template

        if "train" not in raw_datasets:
            raise ValueError("Requires a train dataset")
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


        self.train_dataset = train_dataset; self.dev_dataset = eval_dataset; self.test_dataset = predict_dataset
        print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))

    
    def train_dataloader(self):
        data_collator = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)
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
        data_collator = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)
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
        data_collator = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)
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
        