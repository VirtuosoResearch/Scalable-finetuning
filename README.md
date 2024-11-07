# Scalable Fine-tuning from Multiple Data Sources: A First-Order Approximation Approach
- Authors: [Dongyue Li](https://lidongyue12138.github.io/), [Ziniu Zhang](https://ziniuzhang.github.io/), [Lu Wang](https://web.eecs.umich.edu/~wangluxy/) and [Hongyang R. Zhang](https://www.hongyangzhang.com/)
- Paper: [arXiv]()

![pipline](./gradex.png)


## Overview

This code implements a fast method for **Es**timation of fine-tuning model losses using **Grad**ients (GradEx). Given a list subsets of tasks, this method can estimate the LM fine-tuning losses on the subsets, without repeated model fine-tuning. It trades off the repeated model fine-tuning with solving logistic regression using gradients as features. It can be applied in subset selection problems to perform task/data selection in fine-tuning language models. We provide the code for experiments of chain-of-thought fine-tuning and intruction fine-tuning.

### Requirements

To build up the environment, please run following commands.

```bash
conda create -n gradex python=3.10
conda activate gradex

pip install -r requirements.txt 
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124  # check the correct version for pytorch nightly about CUDA

mkdir ./data
mkdir ./results
mkdir ./external_lightning_logs

python setup.py develop
```

### Data Preparation

**Chain-of-thought fine-tuning.** Please refer to the [reasoning-teacher repository](https://github.com/itsnamgyu/reasoning-teacher) for downloading the chain-of-thought data, including CommonsenseQA and StartegyQA. 

**Instruction fine-tuning**. 

- Alpaca: Please download the data from [this link](https://github.com/HazyResearch/skill-it/blob/main/aux_data/alpaca_final.pkl) and put the pickle file under the `./data/alpaca_data` folder.  


- FLAN v2: Please refer to the [open-instruct repository](https://github.com/allenai/open-instruct) for downloading the FLAN v2 (and COT) instruction fine-tuning data. 

After downloading, save the data set as pickle file under the `./data/alpaca_data` folder.  For example: 

```python
from datasets import load_dataset
import pandas as pd

flan_dataset = load_dataset("json", data_files="./raw_train/tulu_v1_resampled_flan_100k.jsonl")["train"]

def reformat_flan(example):
    prompt = example["inputs"]
    if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
        prompt += "\n"
    completion = example["targets"]
    example['text'] = prompt + completion
    example['skill'] = example['_task_name']
    return example

flan_dataset_df = flan_dataset.map(reformat_flan)
pd.to_pickle(flan_dataset_df, "./flan_dataset.pkl")
```


### Usage

Our algorithm contains the following steps:

1. **Meta training**: Multitask training on all tasks to obtain a meta-initialization. Then, we evaluate and project the gradients of all training samples on the meta-initialization. 
2. **Estimation**: Estimate model fine-tuning performances on a list of task subsets using projected gradients as features in logistic regression. 
3. **Selection**: Using the estimated results to conduct subset selection, including forward stepwise selection and random ensemble. 


#### Meta Training:

This step fine-tune a language model on the combination of all tasks. 

- Use `custom_train_cot.py` to fine-tune LMs on the chain-of-thought data. 

- Use `custom_train_instruction.py` to fine-tune LMs on the instruction fine-tuning data. Use `--train_instruction` to load the FLAN v2 dataset. Without `--train_instruction`, it will load the Alpaca dataset. 

We provide scripts examples under `scripts/meta_training_**.sh`. 

**Evaluating and projecting gradients** on all training samples: 

- For chain-of-thought fine-tuning, use `fast_estimate_compute_gradients_cot.py`. Use `--load_model_dir` to specify a saved checkpoint directory as the base model. Specify `--project_dim` as the number of projections. 

- For instruction fine-tuning, use `fast_estimate_eval_approximation_instruction.py`. Use `--train_instruction` to load the FLAN v2 or Alpaca datasets. Use `--compute_pretrained_outputs` to compute the gradients. The parameters is similar to the file above. 

We provide bash script examples under `scripts/fast_estimate_gradients_**.sh`. These files will save the projection matrix and all projected gradients under a `./gradients/` folder. Please create the folder before usage. 

#### Estimation:

We solve linear regression using the gradients collected above as features to estimate the output of model fine-tuned on a subset of tasks. 

- For chain-of-thought fine-tuning, use `fast_estimate_linear_model_cot.py`. Specify `--save_name` for the file to save the evaluation results of estimated models. Specify `--number_of_subsets` and `--subset_size` to control the number and size of sampled subsets

- For instruction fine-tuning, use `fast_estimate_linear_regression_alpaca.py`.  The parameters is similar to the above. 

We provide bash script examples under `scripts/fast_estimate_logistic_regression_**.sh`. Inside the files, one can modify the subsets collection file under `./sampled_tasks/` to specify the sampled subsets of tasks. Normally, it should be randomly sampled subsets. 

#### Selection:
- Forward stepwise selection: please refer to `utils/fast_estimate_forward_selection.py` to conduct forward selection to select a subset of data. 
- Random ensemble: Please refer to `utils/select_random_ensemble.py` for an example of estimating random ensemble scores. Then, we apply a threshold (or can be viewed as top-k selection) to the scores to select a subset of tasks. 


## Examples

We provide examples for illustrating the use cases of our algorithm:
- Select correct examples from a noisy synthetic dataset: [`examples/example_noisy_synthetic_addition_task.ipynb`](https://github.com/VirtuosoResearch/Scalable-finetuning/blob/main/examples/example_noisy_synthetic_addition_task.ipynb)


## Reference
If you find this repository useful or happen to use it in a research paper, please our work with the following bib information

```
@article{Li2024scalable,
  title={Scalable Fine-tuning from Multiple Data Sources: A First-Order Approximation Approach},
  author={Li, Dongyue and Zhang, Ziniu and Wang, Lu and Zhang, Hongyang R},
  journal={EMNLP Findings},
  year={2024},
}
```
