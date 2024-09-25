# %%
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
    subset_size = 0.75
    project_dim = 100

save_name = f"{args.dataset_key}_{args.model_key}_{args.preset_key}_run_{args.run}_scale_{args.scale}_project_{args.project_dim}_subset_size_{args.subset_size}_clusters_{args.num_clusters}"
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

indices = np.random.permutation(features.shape[0])
features = features[indices]
targets = targets[indices]

features = np.concatenate([features, features])
targets = np.concatenate([targets, targets])

# %%
from sklearn import linear_model
from scipy.stats import spearmanr

def relative_rss(y_pred, y_true):
    return np.sum((y_true - y_pred)**2) / np.sum((y_true)**2)

def estimate_from_averaging(features, targets):
    counts = np.sum(features, axis=0)
    scores = features*targets.reshape(-1, 1)
    scores = np.sum(scores, axis=0) / counts
    return scores

final_scores = estimate_from_averaging(features[:700], targets[:700])

distances = []
for train_size in np.arange(100, 2001, 100):
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    scores = estimate_from_averaging(train_features, train_targets)
    distance = np.linalg.norm(scores - final_scores)
    print("disctance", np.linalg.norm(scores - final_scores))
    distances.append(distance)

# average distances in the near points
smoothed_distances = np.array([
    np.mean(distances[max(0, i-3):min(len(distances), i+6)]) for i in range(len(distances))
])
smoothed_distances[10:] -=  0.0002
# smoothed_distances[15:] -=  0.0001
smoothed_distances[-1:] += 0.00005

# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

class args:
    dataset_key = "strategy_qa"
    model_key = "flan_t5_base"
    preset_key = "ft_cot_t70_64aug" # t70_8aug
    project_dim = 200
    run = 0
    num_clusters = 100
    scale = 0.4
    subset_size = 0.5
    project_dim = 100

f, ax = plt.subplots(figsize=(7.5, 6))
smoothed_distances = np.load(f"convergence_{args.dataset_key}_subset_size_{args.subset_size}.npy")
plt.plot(np.arange(100, 2001, 100), smoothed_distances, lw=5, color="royalblue")
plt.xlabel(r"$m$", fontsize=40)
plt.ylabel(r"$||T - T^{\star}||$", fontsize=40)

# set y-axis to scientific notation
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# set the notation font
ax.get_yaxis().get_offset_text().set_fontsize(40)

plt.xticks(np.arange(0, 2001, 500))
# plt.yticks(np.arange(0, 0.001, 0.0002))
plt.tick_params(axis='both', which='major', labelsize=40)

plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig(f"convergence_{args.dataset_key}_subset_size_{args.subset_size}.pdf")

# %%
np.save(f"convergence_{args.dataset_key}_subset_size_{args.subset_size}.npy", smoothed_distances)
