# %%
import os
import pandas as pd
import numpy as np

class args:
    dataset_key = "multiarith"
    model_key = "flan_t5_base"
    preset_key = "ft_cot_t70_64aug" #
    project_dim = 200
    run = 0
    num_clusters = 100

save_name = f"{args.dataset_key}_{args.model_key}_{args.preset_key}_run_{args.run}_scale_0.4_clusters_{args.num_clusters}"
file_dir = os.path.join("./results/", save_name)
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
    sample_task = [int(task) for task in sample_task if task]

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
from sklearn import linear_model, neural_network, ensemble
from scipy.stats import spearmanr

def relative_rss(y_pred, y_true):
    return np.sum((y_true - y_pred)**2) / np.sum((y_true)**2)

indices = np.random.permutation(features.shape[0])
features = features[indices]
targets = targets[indices]
test_size = 100
train_features, test_features = features[:-test_size], features[-test_size:]
train_targets, test_targets = targets[:-test_size], targets[-test_size:]

mean, std = train_targets.mean(), train_targets.std()
train_targets = (train_targets - mean) / std
test_targets = (test_targets - mean) / std

# %%
corrs = []
for train_size in np.arange(100, 1001, 100):
    tmp_features, tmp_targets = train_features[:train_size], train_targets[:train_size]
    clf = linear_model.Ridge(alpha=1e-2, fit_intercept=False)
    clf.fit(tmp_features, tmp_targets)
    print(f"Train size: {train_size}", 
          "Train Score: ", clf.score(tmp_features, tmp_targets), 
          "Relative RSS: ", relative_rss(clf.predict(tmp_features), tmp_targets),
          "Spearman: ", spearmanr(clf.predict(tmp_features), tmp_targets)[0],
          "Test Score: ", clf.score(test_features, test_targets), 
          "Relative RSS: ", relative_rss(clf.predict(test_features), test_targets),
          "Spearman: ", spearmanr(clf.predict(test_features), test_targets)[0],)
    corrs.append(spearmanr(clf.predict(test_features), test_targets)[0])

# %%
import matplotlib.pyplot as plt

plt.plot(np.arange(100, 2001, 100), corrs)

# %%
# import logistic regression model
from sklearn.linear_model import LogisticRegression
# import f1_score
from sklearn.metrics import f1_score, accuracy_score

indices = np.random.permutation(features.shape[0])
features = features[indices]
targets = targets[indices]
train_features, test_features = features[:-500], features[-500:]
train_targets, test_targets = targets[:-500], targets[-500:]

threshold = 0.0388
train_targets = np.where(train_targets > threshold, 1, 0)
test_targets = np.where(test_targets > threshold, 1, 0)

for train_size in np.arange(100, 2001, 100):
    tmp_features, tmp_targets = train_features[:train_size], train_targets[:train_size]
    clf = LogisticRegression(C=1e-1)
    clf.fit(tmp_features, tmp_targets)
    print(f"Train size: {train_size}", 
          "Train Score: ", accuracy_score(clf.predict(tmp_features), tmp_targets),
          "Test Score: ", accuracy_score(clf.predict(test_features), test_targets))

# 0.6967741935483871
# %%
''' 
Evaluate gradient similarity
'''
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

train_gradients = []
for file in os.listdir(gradients_dir):
    if "train" in file:
        gradients = np.load(f"{gradients_dir}/{file}")
        train_gradients.append(gradients)

train_gradients = np.concatenate(train_gradients, axis=0)

gradient_similarity_scores = (train_gradients * avg_test_gradients.reshape(1, -1)).sum(axis=1)

np.save(f"./scores/{args.dataset_key}_{args.preset_key}_gradient_similarity_scores_{args.model_key}_{args.project_dim}.npy", gradient_similarity_scores)

# %%
from scipy.stats import spearmanr

gradient_similarity_scores = np.load(f"./scores/{args.dataset_key}_{args.preset_key}_dsir_scores.npy")
scores = []
for idx in range(args.num_clusters):
    tmp_idxes = np.load(f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/clusters_{args.num_clusters}/cluster_{idx}.npy") 
    scores.append(gradient_similarity_scores[tmp_idxes].mean())
scores = np.array(scores)

def predict_from_gradient_similarity(scores, feature):
    return (feature * scores.reshape(1, -1)).sum(axis=1)

print("Spearman: ", spearmanr(predict_from_gradient_similarity(scores, test_features), test_targets)[0])
# Gradients:  0.1672165498124423
# DSIR: 0.11632591099893588



# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

# test_pred = test_pred*std + mean
# test_targets = test_targets*std + mean

f, ax = plt.subplots(figsize=(5, 4))
# optimal_mse = 0.0016/std

train_nums = np.arange(100, 2001, 100)
correlation = np.array(
    [0.042674383253412544,
 0.23601583254092982,
 0.2334998861933099,
  0.2416353530920111,
  0.24615969926605276,
  0.3228415870019417,
  0.31652165329863413,
 0.34596132796722366,
 0.34331942177338885,
  0.3569978627817431,
   0.36142229678376306,
  0.3624340244207336,
  0.37337992788000474,
 0.3858020210028872,
   0.38952676837528804,
   0.404258781018526,
 0.40580361758403816,
 0.40580361758403816,
 0.40580361758403816,
 0.40580361758403816]
)
# smooth
correlation = np.concatenate([np.convolve(correlation, np.ones(3)/3, mode='same')[:-3], correlation[-3:]])
ax.plot(train_nums, correlation, lw=4, color = "darkblue",)
# ax.fill_between(
#     train_nums, correlation-0.03, correlation+0.03, 
#     alpha=0.3, color = "darkblue",
#     )

plt.xticks(np.arange(0, 2001, 500), fontsize=28)
# HI
# plt.yticks(np.arange(0., 2.1, 0.5), fontsize=28)
# plt.ylim(-0.1, 1.6)
# LA
plt.yticks(np.arange(0., 0.5, 0.1), fontsize=28)
plt.ylim(0, 0.45)
plt.xlabel(r'$n$', fontsize=36)
plt.ylabel(r'$\mathrm{Spearman~Corr.}$', fontsize=30)
# plt.title(r'$\mathrm{Spearman~Corr.}\mathrm{:}~0.77$', fontsize=30)
# plt.legend(fontsize=20)
plt.grid(ls=':', lw=0.8)
plt.tight_layout()
plt.savefig(f"./notebooks/figures/spearsmanr_multiarith.pdf", format="pdf", dpi=1200)
plt.show()
