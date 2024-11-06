import numpy as np
import pandas as pd
from datasets import Dataset


# Functions for simple input-output format
def format_simple_example(df, idx, add_newline=True, input_columns=["input"], output_columns=["output"]):
    prompt = {}    
    prompt['input'] = df["input"].iloc[idx] + "\n"
    prompt["output"] = ""
    for column in input_columns: # deprecated for training from scratch
        if "index_i" in column:
            prompt["index_i"] = df[column].iloc[idx]
        elif "index_j" in column:
            prompt["index_j"] = df[column].iloc[idx]
        if "step" in column and (not pd.isna(df[column].iloc[idx])):
            prompt["input"] += (str(df[column].iloc[idx]) + "\n")
    for i, column in enumerate(output_columns):
        if pd.isna(df[column].iloc[idx]): 
            prompt[f"column_{column}"] = ""
        else:
            # add output for each column
            prompt[f"column_{column}"] = str(df[column].iloc[idx])
        prompt["output"] += (str(df[column].iloc[idx]) + "\n") if i < len(output_columns) - 1 else (str(df[column].iloc[idx]))
    prompt["output"] += ("\n\n" if add_newline else "")
    return prompt

def generate_simple_algorithm_example(df, idx, k=0, input_columns=["input"], output_columns=["output"]):

    if k == 0:
        return format_simple_example(df, idx, add_newline=False, input_columns=input_columns, output_columns=output_columns)
    
    rng = np.random.default_rng(42)
    indexes = rng.choice(df.shape[0], size=k, replace=False)
    input_prompt = ""
    for i in indexes:
        tmp_prompt = format_simple_example(df, i, input_columns=input_columns, output_columns=output_columns)
        input_prompt = tmp_prompt["input"] + tmp_prompt["output"]
    
    prompt = {}
    prompt['input'] = input_prompt + df["input"].iloc[idx] + "\n"
    prompt["output"] = str(df["output"].iloc[idx])
    return prompt


# Function to split train, val, and test datasets
def load_algorithmic_dataset_with_intermediate_steps(algorithm, data_dir, train_size, valid_size, test_size, num_of_instances=1000000, 
                                                     only_ouptut=False):
    """ Load intermediate steps into a single sequence """    
    file_name = f"./data/{algorithm}/{data_dir}.csv"
    instance_df = pd.read_csv(file_name, index_col=0)

    data_len = min(num_of_instances, instance_df.shape[0])
    rng = np.random.default_rng(42)
    shuffle_idxes = rng.permutation(data_len)
    train_idxes = shuffle_idxes[:train_size]
    valid_idxes = shuffle_idxes[-valid_size-test_size:-test_size]
    test_idxes = shuffle_idxes[-test_size:]

    train_df = instance_df.iloc[train_idxes]
    valid_df = instance_df.iloc[valid_idxes]
    test_df = instance_df.iloc[test_idxes]

    def generate_dataset(instance_df, input_columns=[], output_columns=[]):
        num_of_instances = instance_df.shape[0]
        def gen(input_columns, output_columns):
            for i in range(num_of_instances):
                yield generate_simple_algorithm_example(instance_df, i, k=0,
                        input_columns=input_columns, output_columns=output_columns) # currenly set args.incontext_k to 0
        dataset = Dataset.from_generator(
            generator=gen, cache_dir="./data/cache_single_prompt",
            gen_kwargs={"input_columns": input_columns, "output_columns": output_columns})
        return dataset

    column_names = ([f"step_{i}" for i in range(min(len(instance_df.columns), 30)) if f"step_{i}" in instance_df.columns] + ["output"])\
         if not only_ouptut else ["output"]

    train_dataset = generate_dataset(train_df, input_columns=["input"], output_columns=column_names[:])
    valid_dataset = generate_dataset(valid_df, input_columns=["input"], output_columns=column_names[:]) # include all previous steps if concatenate_steps
    test_dataset = generate_dataset(test_df, input_columns=["input"], output_columns=column_names[:]) # include all previous steps if concatenate_steps
    print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")


    return train_dataset, valid_dataset, test_dataset, column_names
