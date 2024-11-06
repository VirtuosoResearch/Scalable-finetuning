import string
import re
import json
import sys
import os
import argparse
import logging
from collections import Counter
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match_score(prediction, ground_truth, xlingual=False, indices = None):
    normalize_pred = normalize_answer(prediction).split()
    normalize_truth = normalize_answer(ground_truth).split()
    count = 0; min_length = min(len(normalize_pred), len(normalize_truth))
    if indices is None: indices = list(range(min_length))
    if len(indices) == 0: return 1 
    for i in indices:
        if i >= min_length: continue
        if normalize_pred[i] == normalize_truth[i]:
            count += 1
    # return (normalize_answer(prediction) == normalize_answer(ground_truth))
    return (count / len(indices)) if min_length != 0 else 0


def edit_distance(list1, list2):
    m, n = len(list1), len(list2)
    
    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table in a bottom-up manner
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If the first string is empty, insert all characters of the second
            elif j == 0:
                dp[i][j] = i  # If the second string is empty, remove all characters of the first
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],        # Remove
                                  dp[i][j - 1],        # Insert
                                  dp[i - 1][j - 1])    # Replace
    
    return dp[m][n]

def edit_distance_score(prediction, ground_truth, xlingual=False, indices = None):
    if indices is None: 
        normalize_pred = normalize_answer(prediction).split()
        normalize_truth = normalize_answer(ground_truth).split()
    else:
        normalize_pred = [normalize_answer(prediction).split()[i] if i < len(normalize_answer(prediction).split()) else "" for i in indices]
        normalize_truth = [normalize_answer(ground_truth).split()[i] for i in indices]
    
    # Compute the edit distance between normalized prediction and truth
    distance = edit_distance(normalize_pred, normalize_truth)
        
    return distance


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False, indices=None, **kwargs):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual, indices=indices, **kwargs)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

''' Deprecated '''

def exact_match_substring(prediction, ground_truth, xlingual=False,
                        keyword="index", shift_idxes = [1, 3]):
    normalize_pred = normalize_answer(prediction).split()
    normalize_truth = normalize_answer(ground_truth).split()
    correct = 0; sum = 0
    min_length = min(len(normalize_pred), len(normalize_truth))
    for i in range(min_length):
        if normalize_truth[i] == keyword:
            for shift in shift_idxes:
                if normalize_pred[i+shift] == normalize_truth[i+shift]:
                    correct += 1
                sum += 1
    assert sum != 0
    return correct/sum

def compute_metrics_for_sorting(predictions, references, xlingual=False, length=5):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, compare_accuracy, swap_accuracy, inter_accuracy, answer_accuracy = 0, 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        compare_accuracy += metric_max_over_ground_truths(
            exact_match_substring, prediction=pred, ground_truths=gold, xlingual=xlingual,
            keyword="index", shift_idxes=[1, 3]
        )
        swap_accuracy += metric_max_over_ground_truths(
            exact_match_substring, prediction=pred, ground_truths=gold, xlingual=xlingual,
            keyword="swap", shift_idxes=[1]
        )
        inter_accuracy += metric_max_over_ground_truths(
            exact_match_substring, prediction=pred, ground_truths=gold, xlingual=xlingual,
            keyword="array", shift_idxes=[i for i in range(1, length+1)]
        )
        answer_accuracy += metric_max_over_ground_truths(
            exact_match_substring, prediction=pred, ground_truths=gold, xlingual=xlingual,
            keyword="answer", shift_idxes=[i for i in range(1, length+1)]
        )
    exact_match = 100.0 * exact_match / len(references)
    compare_accuracy = 100.0 * compare_accuracy / len(references)
    swap_accuracy = 100.0 * swap_accuracy / len(references)
    inter_accuracy = 100.0 * inter_accuracy / len(references)
    answer_accuracy = 100.0 * answer_accuracy / len(references)
    
    metrics = {"exact_match": exact_match, "compare_accuracy": compare_accuracy, 
               "swap_accuracy": swap_accuracy, "inter_accuracy": inter_accuracy,
               "answer_accuracy": answer_accuracy}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def accuracy_score(prediction, ground_truth, xlingual=False, shift=0):
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) <= shift:
        return 0
    return (prediction[shift] == ground_truth[shift])

def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def rouge2_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge2'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge2"].fmeasure

def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    accuracy, exact_match, rouge1, rougeL, rouge2 = 0, 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        accuracy += metric_max_over_ground_truths(
            accuracy_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge2 += metric_max_over_ground_truths(
            rouge2_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    accuracy = 100.0 * accuracy / len(references)
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    rouge2 = 100.0 * rouge2 / len(references)
    metrics = {"accuracy": accuracy, "exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL, "rouge2": rouge2}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--predictions", required=True, help="Path to predictions file.")
#     parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
#         help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
#     )
#     parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
#     parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
#     return parser.parse_args()


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        if self._data.counts[key] > 0:
            self._data.average[key] = self._data.total[key] / self._data.counts[key]
        else:
            self._data.average[key] = 0

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def compute_accuracy(predictions, references, task_indices=None):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    xlingual = False
    if task_indices is None:
        task_indices = [None] * len(predictions)
        metrics = {"accuracy": 0, "edit_distance": 0}
    else:
        metrics = {f"{task_name}_accuracy": 0 for task_name in task_indices[0].keys()}
        metrics.update({f"{task_name}_edit_distance": 0 for task_name in task_indices[0].keys()})
        metrics.update({f"{task_name}_num_samples": 0 for task_name in task_indices[0].keys()})
    
    for pred, gold, task_idx in zip(predictions, references, task_indices):
        assert isinstance(gold, list)
        if task_idx is None:
            metrics["accuracy"] += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            metrics["edit_distance"] += metric_max_over_ground_truths(
                edit_distance_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
        else:
            # task_idx is dict
            for task_name, idx in task_idx.items():
                if len(idx) == 0:
                    continue
                tmp_accuracy = metric_max_over_ground_truths(
                    exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual, indices = idx
                )
                tmp_edit_distance = metric_max_over_ground_truths(
                    edit_distance_score, prediction=pred, ground_truths=gold, xlingual=xlingual, indices = idx
                )
                metrics[f"{task_name}_accuracy"] += tmp_accuracy
                metrics[f"{task_name}_edit_distance"] += tmp_edit_distance
                metrics[f"{task_name}_num_samples"] += 1
    
    if task_indices is None:
        metrics["accuracy"] = 100.0 * metrics["accuracy"] / len(references)
        metrics["edit_distance"] = metrics["edit_distance"] / len(references)
    else:
        for key, val in metrics.items():
            if "accuracy" in key:
                task_name = "_".join(key.split("_")[:-1])
                metrics[key] = (100.0 * val / metrics[f"{task_name}_num_samples"]) if metrics[f"{task_name}_num_samples"] > 0 else 0
            elif "edit_distance" in key:
                task_name = "_".join(key.split("_")[:-2])
                metrics[key] = (val / metrics[f"{task_name}_num_samples"]) if metrics[f"{task_name}_num_samples"] > 0 else 0
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics
    