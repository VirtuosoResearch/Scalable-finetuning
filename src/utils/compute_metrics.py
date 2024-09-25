import string
import re
import json
import sys
import os
import argparse
import logging
from collections import Counter
# from rouge import rouge_scorer
from transformers import AutoTokenizer


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


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def accuracy_score(prediction, ground_truth, xlingual=False):
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) == 0:
        return 0
    return (prediction[0] == ground_truth[0])

# def rouge1_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
#     else:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rouge1"].fmeasure


# def rougeL_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
#     else:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rougeL"].fmeasure

# def rouge2_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rouge2'], tokenizer=xlingual_tokenizer) 
#     else:
#         scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rouge2"].fmeasure

def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    accuracy, exact_match, rouge1, rougeL, rouge2, count = 0, 0, 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        if len(normalize_answer(gold).split()) == 0:
            continue

        accuracy += accuracy_score(pred, gold)
        exact_match += exact_match_score(pred, gold)
        count += 1
        # rouge1 += metric_max_over_ground_truths(
        #     rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        # )
        # rougeL += metric_max_over_ground_truths(
        #     rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        # )
        # rouge2 += metric_max_over_ground_truths(
        #     rouge2_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        # )
    accuracy = 100.0 * accuracy / count
    exact_match = 100.0 * exact_match / count
    # rouge1 = 100.0 * rouge1 / len(references)
    # rougeL = 100.0 * rougeL / len(references)
    # rouge2 = 100.0 * rouge2 / len(references)
    metrics = {"accuracy": accuracy, "exact_match": exact_match} # "rouge1": rouge1, "rougeL": rougeL, "rouge2": rouge2
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
