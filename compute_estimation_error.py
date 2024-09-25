# %%
import pandas as pd 
import numpy as np

# pretrained_estimate_df = pd.read_csv("./results/Alpaca_EleutherAI-gpt-neo-125m_lora_r_4_dim_200_run_0_pretrained/results.csv", index_col=0)
# finetuned_df = pd.read_csv("./results/Alpaca_EleutherAI-gpt-neo-125M_lora_r_4/results.csv", index_col=0)
criterion = "accuracy"
pretrained_estimate_df = pd.read_csv("./results/strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_project_200_clusters_100_forward_selection/results.csv", index_col=0)
finetuned_df = pd.read_csv("./results/strategy_qa_flan_t5_base_ft_cot_t70_64aug_lora_r_16_forward_selection/results.csv", index_col=0)


# %%
data_indices = finetuned_df["Data indices"].values
true_values = []
estimated_values = []
for line in data_indices:
    true_values.append(finetuned_df[finetuned_df["Data indices"] == line][criterion].values[0])
    try:
        estimated_values.append(pretrained_estimate_df[pretrained_estimate_df["Data indices"] == line][criterion].values[0])
    except:
        true_values.pop()
true_values = np.array(true_values)
estimated_values = np.array(estimated_values)
# %%
error = np.square(np.abs(true_values - estimated_values))/np.square(true_values)
print(error.mean())
# %%
