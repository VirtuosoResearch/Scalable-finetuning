# %%
import os
import numpy as np

class args:
    dataset_key = "strategy_qa"
    model_key = "flan_t5_base"
    preset_key = "ft_cot_t70_64aug"
    project_dim = 200
    run = 0

gradients_dir = f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/run_{args.run}"

# load test gradients 
avg_test_gradients = None; count = 0
for file in os.listdir(gradients_dir):
    if "test" in file:
        gradients = np.load(f"{gradients_dir}/{file}")
        if avg_test_gradients is None:
            avg_test_gradients = gradients.sum(axis=0)
        else:
            avg_test_gradients += gradients.sum(axis=0)
        count += gradients.shape[0]
avg_test_gradients /= count
# %%
train_gradients = []
for file in os.listdir(gradients_dir):
    if "train" in file:
        gradients = np.load(f"{gradients_dir}/{file}")
        train_gradients.append(gradients)

train_gradients = np.concatenate(train_gradients, axis=0)
# %%
scores = (train_gradients * avg_test_gradients.reshape(1, -1)).sum(axis=1)
# %%
index_dir = "./data_indices"
indices = np.argsort(scores)
for ratio in np.arange(0.7, 1.0, 0.05):
    num_indices = int(ratio * len(indices))
    with open(f"{index_dir}/{args.dataset_key}_{args.preset_key}_gradients_ratio_{int(ratio*100)}.txt", "w") as f:
        f.write(" ".join([str(idx) for idx in indices[-num_indices:]]))

# %%
import numpy as np
index_dir = "./data_indices"
indices = np.random.permutation(len(scores))
for ratio in np.arange(0.9, 0.4, -0.1):
    num_indices = int(ratio * len(indices))
    with open(f"{index_dir}/{args.dataset_key}_{args.preset_key}_random_ratio_{ratio}.txt", "w") as f:
        f.write(" ".join([str(idx) for idx in indices[-num_indices:]]))