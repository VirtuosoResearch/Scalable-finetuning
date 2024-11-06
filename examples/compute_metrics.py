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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


# if __name__ == "__main__":
    # args = parse_args()
    # with open(args.predictions) as fin:
    #     examples = [json.loads(l) for l in fin]

    # predictions = [e["prediction"] for e in examples]
    # references = [e["Instance"]["output"] for e in examples]
    # tasks = []
    # for e in examples:
    #     if e["Task"] == "task121_atomic_question_rewriting":
    #         e["Task"] = "task121_zest_question_rewriting"
    #     tasks.append(e["Task"])

    # results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
    # print("======== Overall Metrics ========")
    # print("all_rougeL", results["rougeL"])
    # print("all_EM", results["exact_match"])
    # print()
    
    # category_metrics = [
    #     ("Textual Entailment", "exact_match"),
    #     ("Cause Effect Classification", "exact_match"),
    #     ("Coreference Resolution", "exact_match"),
    #     ("Dialogue Act Recognition", "exact_match"),
    #     ("Answerability Classification", "exact_match"),
    #     ("Word Analogy", "exact_match"),
    #     ("Overlap Extraction", "rougeL"),
    #     ("Keyword Tagging", "rougeL"),
    #     ("Question Rewriting", "rougeL"),
    #     ("Title Generation", "rougeL"),
    #     ("Data to Text", "rougeL"),
    #     ("Grammar Error Correction", "rougeL"),
    # ]
    # category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

    # if args.compute_per_category_metrics:
    #     print("======== Metrics per category ========")
    #     task_category = {}
    #     for task in set(tasks):
    #         with open(os.path.join("./data/tasks/", task+".json")) as fin:
    #             task_data = json.load(fin)
    #             task_category[task] = "_".join(task_data["Categories"][0].lower().split())
    #     categories = [task_category[e["Task"]] for e in examples] 
    #     results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
    #     for category, metric in category_metrics.items():
    #         # category = "_".join(category.lower().split())
    #         if f"{metric}_for_{category}" in results:
    #             print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
    #     print()
            
    # if args.compute_per_task_metrics:
    #     print("======== Metrics per task ========")
    #     results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
    #     for task in sorted(list(set(tasks))):
    #         category = task_category[task]
    #         metric = category_metrics[category]
    #         print(task, results_by_task[f"{metric}_for_{task}"])
    #     print()