# %%
import pandas as pd

true_performance_df = pd.read_csv('../results/strategy_qa_flan_t5_base_ft_cot_t70_64aug_lora_r_16_true_performance/results.csv', index_col=0)
estimated_performance_df = pd.read_csv('../results/strategy_qa_flan_t5_base_ft_cot_t70_64aug_run_0_scale_0.4_clusters_100/results.csv', index_col=0)


# %%
true_performances = []
subsets = true_performance_df["Data indices"]
for subset in subsets:
    true_performances.append(true_performance_df[true_performance_df['Data indices']==subset]["accuracy"].values[0])

estimated_performances = []
for subset in subsets:
    if subset in estimated_performance_df["Data indices"].values:
        estimated_performances.append(estimated_performance_df[estimated_performance_df['Data indices']==subset]["accuracy"].values[0])
# %%
import numpy as np

estimated_performances = np.array(estimated_performances)
true_performances = np.array(true_performances)

diff = np.square(estimated_performances - true_performances).sum()
print(diff/np.square(true_performances).sum())