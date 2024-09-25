import argparse
import logging
import os
import numpy as np

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from src.custom.model import Model

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

import cvxpy as cp
import numpy as np

def run_sdp_clustering(task_affinities, k, use_exp=True, temperature=1.0):
    if use_exp:
        task_affinities = np.exp(task_affinities/temperature)
    def sdp_clustering(T, k):
        n = T.shape[0]

        A = []
        b = []
        # first constraint 
        A.append(np.eye(n))
        b.append(k)

        # second constraint
        for i in range(n):
            tmp_A = np.zeros((n, n))
            tmp_A[:, i] = 1
            A.append(tmp_A)
            b.append(1)

        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((n,n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0, X>=0]
        constraints += [
            cp.trace(A[i] @ X) == b[i] for i in range(len(A))
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(T @ X)),
                        constraints)
        prob.solve()

        # Print result.
        print("The optimal value is", prob.value)
        X_final = X.value
        X_final = X_final > 1/n
        return X_final, X.value

    maximum = np.max(task_affinities)
    X_final, X_value = sdp_clustering(maximum-task_affinities, k)

    # generate cluster labels
    assignment = {}; cluster_idx = 0; assigned_before = np.zeros(X_final.shape[0])
    for i in range(X_final.shape[0]):
        assigned_count = 0
        for j in range(i, X_final.shape[1]):
            if X_final[i, j] and assigned_before[j] == 0:
                if assigned_before[i] == 0: 
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(i) 
                    else:
                        assignment[cluster_idx] = [i]
                    assigned_count += 1
                    assigned_before[i] = 1
                if assigned_before[j] == 0:
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(j) 
                    else:
                        assignment[cluster_idx] = [j]
                    assigned_count += 1
                    assigned_before[j] = 1
        if assigned_count > 0:
            cluster_idx += 1

    for cluster_idx in assignment:
        print(" ".join([str(idx) for idx in assignment[cluster_idx]]))
    return assignment

def main(args):
    gradient_dir = f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/run_{args.run}"
    gradients = []
    for file in os.listdir(gradient_dir):
        if file.startswith("train"):
            gradients.append(np.load(os.path.join(gradient_dir, file)))
    gradients = np.concatenate(gradients, axis=0)

    # gradient_similarities = np.dot(gradients, gradients.T)

    # Estimate cluster gradients
    # clustering = SpectralClustering(n_clusters=args.num_clusters, 
    #                                 affinity="rbf", 
    #                                 n_init=10).fit(gradient_similarities)
    clustering = AgglomerativeClustering(n_clusters=args.num_clusters, 
                                    metric="cosine",
                                    linkage="average").fit(gradients)
    assignments = clustering.labels_
    # assignments_dict = run_sdp_clustering(gradient_similarities, args.num_clusters, use_exp=True, temperature=1.0)
    # assignments = np.zeros(gradients.shape[0])
    # for cluster_idx in assignments_dict:
    #     assignments[assignments_dict[cluster_idx]] = cluster_idx

    cluster_dir = f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/clusters_{args.num_clusters}"

    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    # write the gradients as the assignment of labels
    for i in range(args.num_clusters):
        cur_cluster = np.nonzero(assignments == i)[0]
        np.save(os.path.join(cluster_dir, f"cluster_{i}.npy"), cur_cluster)
        print(cur_cluster.shape)

    # # %%
    # def computer_inter_cluster_density(features, assignments):
    #     n_clusters = len(np.unique(assignments))
    #     cluster_density = np.zeros(n_clusters)
    #     for i in range(n_clusters):
    #         cur_cluster = features[assignments == i]
    #         # computer pairwise distance
    #         for j in range(len(cur_cluster)):
    #             cluster_density[i] += np.linalg.norm(cur_cluster - cur_cluster[j].reshape(1, -1), axis=1).sum()
    #         cluster_density[i] /= len(cur_cluster)
    #     return cluster_density

    # density = computer_inter_cluster_density(gradients, assignments)
    # print(np.mean(density))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="multiarith")
    parser.add_argument("--model_key", type=str, default="flan_t5_base")
    parser.add_argument("--train_key", type=str, default="ft_cot")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--preset_key", type=str, default="ft_cot_t70_64aug")
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disable_checkpointing", action="store_true")

    parser.add_argument("--project_dim", type=int, default=200)
    parser.add_argument("--run", type=int, default=0)

    parser.add_argument("--num_clusters", type=int, default=200)
    
    args = parser.parse_args()
    main(args)