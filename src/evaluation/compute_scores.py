import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def compute_final_error(split_errors, alpha=0.5):
    f = split_errors["forecast"]
    r = split_errors["reconstruction"]

    if r is None:
        return f  # GDN case
    return alpha * f + (1 - alpha) * r  # MTAD-GAT case
    
def load_errors_all_splits(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["preprocessing"]["dataset_name"]
    eval_results_folder = f"{project_root_dir}/eval_results/{dataset_name}"
    model_name = config["evaluation"]["model"]

    errors_folder = f"{eval_results_folder}/{model_name}/errors"
    splits = ["train", "val", "actor2_test", "actor1_test"]

    errors = {}

    for split in splits:
        path = os.path.join(errors_folder, f"{split}.npz")
        data = np.load(path)
        errors[split] = compute_final_error(data, alpha=0.5)

    return errors

# def normalize(train_scores, scores):
#     min_v = train_scores.min()
#     max_v = train_scores.max()
#     return (scores - min_v) / (max_v - min_v + 1e-8)

def normalize_errors_all_splits(errors):
    """
    Normalize errors using train statistics only
    """
    median = np.median(errors["train"], axis=0)
    iqr = np.percentile(errors["train"], 75, axis=0) - np.percentile(errors["train"], 25, axis=0)
    # stabilize
    iqr = np.maximum(iqr, 0.05)

    # norm and clip
    train_norm = np.abs((errors["train"] - median) / iqr)
    val_norm = np.abs((errors["val"] - median) / iqr)
    actor2_test_norm = np.abs((errors["actor2_test"] - median) / iqr)
    actor1_test_norm = np.abs((errors["actor1_test"] - median) / iqr)
    
    # clip
    errors["train"] = np.clip(train_norm, 0, 10)
    errors["val"] = np.clip(val_norm, 0, 10)
    errors["actor2_test"] = np.clip(actor2_test_norm, 0, 10)
    errors["actor1_test"] = np.clip(actor1_test_norm, 0, 10)

    return errors, median, iqr

def compute_scores(norm_errors, iqr, topk_ratio=0.4):
    n_sensors = norm_errors.shape[1]
    k = max(1, int(topk_ratio * n_sensors))

    # weight stable sensors more
    weights = 1 / (iqr + 1e-6)
    weights = weights / np.sum(weights)

    # top-k selection
    idx = np.argsort(norm_errors, axis=1)[:, -k:]

    topk_vals = np.take_along_axis(norm_errors, idx, axis=1)
    topk_weights = np.take_along_axis(weights[None, :], idx, axis=1)

    scores = np.sum(topk_vals * topk_weights, axis=1)

    # log compression
    scores = np.log1p(scores)

    return scores

def compute_scores_all_splits(errors_norm, iqr):
    train_scores = compute_scores(errors_norm["train"], iqr)
    val_scores = compute_scores(errors_norm["val"], iqr)
    actor2_test_scores = compute_scores(errors_norm["actor2_test"], iqr)
    actor1_test_scores = compute_scores(errors_norm["actor1_test"], iqr)

    # soomthing
    train_scores = smooth_scores(train_scores)
    val_scores = smooth_scores(val_scores)
    actor2_test_scores = smooth_scores(actor2_test_scores)
    actor1_test_scores = smooth_scores(actor1_test_scores)

    return {
        "train": train_scores,
        "val": val_scores,
        "actor2_test": actor2_test_scores,
        "actor1_test": actor1_test_scores,
    }

def compute_segment_metrics(pred, start, end):
    segment = pred[start:end]

    # SDR
    SDR = int(np.any(segment))

    # Coverage
    coverage = np.mean(segment)

    # Delay
    if SDR:
        delay = np.argmax(segment)
    else:
        delay = np.inf

    return SDR, coverage, delay

def segment_evaluation(config):
    errors = load_errors_all_splits(config)
    errors_norm, _, iqr = normalize_errors_all_splits(errors)
    scores = compute_scores_all_splits(errors_norm, iqr)
    
    threshold = np.percentile(scores["train"], 95)
    # threshold = 0.5

    start_actor2 = len(scores["train"])
    end_actor2 = start_actor2 + len(scores["actor2_test"])

    full_scores = np.concatenate([
        scores["train"],
        scores["actor2_test"],
        # scores["actor1_test"]
    ])

    pred = (full_scores > threshold).astype(int)

    SDR, coverage, delay = compute_segment_metrics(pred, start_actor2, end_actor2)

    print("SDR: ", SDR, "coverage: ", coverage, "delay: ", delay)

    normal_mask = np.ones_like(full_scores, dtype=bool)
    normal_mask[start_actor2:end_actor2] = False

    fp_rate = np.mean(pred[normal_mask])
    print("FPR: ", fp_rate)

    # plt.figure(figsize=(14,4))
    # plt.plot(full_scores)
    # plt.axhline(threshold)
    # plt.axvspan(start_actor2, end_actor2, alpha=0.2)
    # plt.title("Full Timeline")
    # plt.show()



def smooth_scores(scores, window=5):
    return pd.Series(scores).rolling(window=window, center=True).mean().fillna(method="bfill").fillna(method="ffill").values

def evalutation_pipeline(config):
    errors = load_errors_all_splits(config)
    errors_norm, _, iqr = normalize_errors_all_splits(errors)
    scores = compute_scores_all_splits(errors_norm, iqr)
    
    threshold = np.percentile(scores["train"], 95)
    # threshold = 0.5
    
    segment_evaluation(scores, threshold)