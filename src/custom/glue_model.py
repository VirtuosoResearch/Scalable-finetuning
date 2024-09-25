import copy
import json
import logging
from typing import List, Dict
import wandb
from collections import defaultdict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import PreTrainedTokenizerBase
import numpy as np
from utils.sam import SAM
import os
from utils.compute_metrics import compute_metrics
from sklearn.metrics import accuracy_score, f1_score

class GLUEModel(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                lr=3e-4, truncate_early=True, max_length=1024, weight_decay=1e-4, use_wandb=False, answer_choices = None,
                optimizer="adamw", generate_output=False):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.lr = lr
        self.max_length = max_length
        self.truncate_early = truncate_early
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.validation_step_outputs = []
        self.answer_choices = answer_choices
        self.optimizer = optimizer
        self.generate_output = generate_output

    def get_trainable_parameters(self):
        return [param for name, param in self.model.named_parameters()\
                if (name in self.param_names) and (not any([key in name for key in self.removing_keys]))]

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            # Save a checkpoint of the model
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', 'ckpt.pt')
            self.trainer.save_checkpoint(ckpt_path)
        return super().on_validation_end()

    def training_step(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        
        loss = self.model(**kwargs)["loss"]
        if self.use_wandb:
            wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """        
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        forward_output = self.model(**kwargs)

        if self.model_type == "encoder_decoder":
            output = self.model.generate(batch["input_ids"], max_length=self.max_length).detach()
            # TODO: not supporting encoder_decoder for now
        elif self.model_type == "decoder":
            ''' Compute the logits of the labels '''
            is_label_mask = batch["labels"][:, 1:].contiguous() != -100
            logits = forward_output["logits"][:, :-1].contiguous()
            preds = logits[is_label_mask]
            preds = preds[:, self.answer_choices]
            preds = torch.argmax(preds, dim=-1).cpu().numpy()

            labels = batch["labels"][:, 1:][is_label_mask]
            for idx, label in enumerate(self.answer_choices):
                labels[labels == label] = idx
            labels = labels.cpu().numpy()
            
            ''' Generate the labels '''
            if self.generate_output:
                # Remove labels in inputs_ids
                is_label_mask = batch["labels"] != -100
                batch["input_ids"][is_label_mask] = self.tokenizer.pad_token_id
                is_input_mask = batch["input_ids"] != self.tokenizer.pad_token_id
                batch["attention_mask"][is_label_mask] = 0

                # convert to left padding
                inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                self.tokenizer.padding_side = 'right'
                inputs = self.transfer_batch_to_device(inputs, self.device, batch_idx)

                output = self.model.generate(**inputs, max_length=self.max_length+1,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id).detach()
                input_len = inputs["input_ids"].shape[1]
                output[:, :input_len] = self.tokenizer.pad_token_id
            else:
                output = None
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        output_dict = {
            "loss": forward_output['loss'],
            "label": batch["labels"],
            "output": output,
            "pred_ids": preds,
            "label_ids": labels,
        }
        self.validation_step_outputs.append(output_dict)
        return output_dict
    
    def on_validation_epoch_end(self) -> None:
        """
        Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
        log validation metrics.

        Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
        """
        outputs = self.validation_step_outputs
        losses = [output["loss"] for output in outputs]
        losses = torch.stack(losses)
        losses = losses[torch.isnan(losses) == False]
        summary = {"loss": losses.mean().item()}
        summary["accuracy_score"] = 0; summary["f1_score"] = 0

        counts = 0
        for batch in outputs:
            if self.generate_output:
                labels = batch["label"]; labels[labels == -100] = self.tokenizer.pad_token_id
                output = batch["output"]; output[output == -100] = self.tokenizer.pad_token_id
            
                label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                output_str = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                metrics = compute_metrics(output_str, label_str)
                
                for key, value in metrics.items():
                    if key not in summary:
                        summary[key] = 0
                    summary[key] += value*len(batch["label_ids"])
            
            if len(batch["label_ids"]) == 0:
                continue
            summary["accuracy_score"] += accuracy_score(batch["label_ids"], batch["pred_ids"])*len(batch["label_ids"])*100
            summary["f1_score"] += f1_score(batch["label_ids"], batch["pred_ids"], average="macro")*len(batch["label_ids"])*100
            counts += len(batch["label_ids"])
        
        for key in summary:
            if key != "loss":
                summary[key] = (summary[key]/counts) if counts > 0 else 0

        # Log metrics
        logging.info(summary)
        if summary:
            for key, value in summary.items():
                if key == "accuracy":
                    self.log(key, value, prog_bar=True, logger=True)
                else:
                    if self.use_wandb:
                        wandb.log({f"val_{key}": value})
                    self.log(key, value, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()
        return summary

    def forward(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = self.model(**kwargs)
        return outputs

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            import bitsandbytes as bnb
            optimizer_dict = {
                "adamw_8bit": bnb.optim.AdamW8bit,
                "paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
                "paged_adamw_32bit": bnb.optim.PagedAdamW32bit,
            }

            optimizer = optimizer_dict[self.optimizer](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # force embedding layers to use 32 bit for numerical stability
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
        return optimizer