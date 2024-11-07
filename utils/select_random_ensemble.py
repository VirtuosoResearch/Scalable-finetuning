# %%
# To run this file, one needs to conduct random sampling on the subsets first and has a csv file recording the results
import os
import pandas as pd
import numpy as np

class args:
    dataset_key = "strategy_qa"
    model_key = "flan_t5_base"
    preset_key = "ft_cot_t70_64aug" # t70_8aug
    project_dim = 200
    run = 0
    num_clusters = 100
    scale = 0.4

save_name = f"{args.dataset_key}_{args.model_key}_{args.preset_key}_run_{args.run}_scale_{args.scale}_clusters_{args.num_clusters}"
file_dir = os.path.join("../results/", save_name)
file_name = os.path.join(file_dir, "results.csv")

result_df = pd.read_csv(file_name, index_col=0)
print(len(result_df))
# %%
num_task = args.num_clusters
target_metric = f"accuracy"
sampled_tasks = result_df["Data indices"].values

features = []
targets = []
for i, subsample in enumerate(sampled_tasks):
    # convert subsample from str to list
    sample_task = subsample.strip('][').split(' ')
    sample_task = [int(task) for task in sample_task if (task and int(task)<num_task)]

    sample_feature = np.zeros(shape=(1, num_task))
    sample_feature[0, sample_task] = 1
    tmp_target = result_df[result_df["Data indices"] == subsample][target_metric].values[0]
    if np.isnan(tmp_target):
        continue
    features.append(sample_feature)
    targets.append(tmp_target)

features = np.concatenate(features, axis=0)
targets = np.array(targets)

# %%
def estimate_from_averaging(features, targets):
    counts = np.sum(features, axis=0)
    scores = features*targets.reshape(-1, 1)
    scores = np.sum(scores, axis=0) / counts
    return scores

args.num_subsets = 800
subset_features = features[:args.num_subsets]
subset_targets = targets[:args.num_subsets]
scores = estimate_from_averaging(subset_features, subset_targets)

index_dir = "../data_indices"
indices = np.argsort(scores)
for ratio in np.arange(70, 100, 5):
    num_indices = int(ratio/100 * len(indices))
    data_idxes = []
    for idx in indices[-num_indices:]:
        tmp_idxes = np.load(f"../gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/clusters_{args.num_clusters}/cluster_{idx}.npy") 
        data_idxes.append(tmp_idxes)
    # tmp_idxes = np.load(f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/clusters_{args.num_clusters}/cluster_10.npy") 
    # data_idxes.append(tmp_idxes)
    data_idxes = np.concatenate(data_idxes)
    data_idxes.sort()
    with open(f"{index_dir}/{args.dataset_key}_{args.preset_key}_random_ensemble_ratio_{int(ratio)}_clusters_{args.num_clusters}_num_subsets_{int(args.num_subsets)}.txt", "w") as f:
        f.write(" ".join([str(idx) for idx in data_idxes]))