import numpy as np
import os
import yaml
import numpy as np
from scipy.stats import genpareto

def compute_metrics(scores, config):
    """
    Detection rate (Actor2) Expect: HIGH (~1.0)
    False positive rate (Actor1 Test) LOW (~0.05)
    """
    eval_results_folder = config["evaluation"]["eval_results_folder"]
    model_name = config["evaluation"]["model"]
    eval_results_per_model = f"{eval_results_folder}/{model_name}"
    
    threshold_percentile = config["evaluation"]["threshold_percentile"]

    # --- Threshold ---
    threshold = np.percentile(scores["train"], threshold_percentile)
    # actor2_test_threshold = np.percentile(scores["actor2_test"], threshold_percentile)
    # actor1_test_threshold = np.percentile(scores["actor1_test"], threshold_percentile)
    # stat, p = ks_2samp(scores["train"], scores["actor2_test"])

    threshold = 0.2
    detection_rate = np.mean(scores["actor2_test"] > threshold)
    false_positive_rate = np.mean(scores["actor1_test"] > threshold)
    print(detection_rate, false_positive_rate)
    with open(os.path.join(f"{eval_results_per_model}/metrics.yaml"), "w") as f:
         yaml.dump({"detection_rate": float(detection_rate)}, f)
         yaml.dump({"false_positive_rate": float(false_positive_rate)}, f)


def compute_anomaly_scores(config):
    eval_results_folder = config["evaluation"]["eval_results_folder"]

    # --- Load errors ---
    train_errors = np.load(f"{eval_results_folder}/errors/train_errors.npy")
    val_errors = np.load(f"{eval_results_folder}/errors/val_errors.npy")
    actor2_test_errors = np.load(f"{eval_results_folder}/errors/actor2_test_errors.npy")
    actor1_test_errors = np.load(f"{eval_results_folder}/errors/actor1_test_errors.npy")

    # Robust Stats
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6  # avoid division by zero

    # --- Normalize ---
    train_norm = (train_errors - median) / iqr
    val_norm = (val_errors - median) / iqr
    actor2_test_norm = (actor2_test_errors - median) / iqr
    actor1_test_norm = (actor1_test_errors - median) / iqr

    train_scores = np.mean(train_norm, axis=1)
    val_scores = np.mean(val_norm, axis=1)
    actor2_test_scores = np.mean(actor2_test_norm, axis=1)
    actor1_test_scores = np.mean(actor1_test_norm, axis=1)
    
    scores = {
        "train": train_scores,
        "val": val_scores,
        "actor2_test": actor2_test_scores,
        "actor1_test": actor1_test_scores
    }

    return scores

def fit_pot_threshold(train_scores, q=0.98, alpha=1e-3):
    """
    Fit POT (Peaks Over Threshold)

    Args:
        train_scores: np.array (normal training scores)
        q: initial threshold quantile (e.g., 0.98)
        alpha: risk level (smaller = stricter threshold)

    Returns:
        final_threshold
    """

    train_scores = np.asarray(train_scores)

    # Step 1: initial threshold u
    u = np.quantile(train_scores, q)

    # Step 2: excesses over threshold
    excesses = train_scores[train_scores > u] - u

    if len(excesses) < 10:
        raise ValueError("Not enough tail samples for POT. Increase dataset or lower q.")

    # Step 3: fit GPD
    # shape (xi), loc, scale (beta)
    xi, loc, beta = genpareto.fit(excesses, floc=0)

    # Step 4: compute final threshold τ
    n = len(train_scores)
    nu = len(excesses)

    # POT formula
    tau = u + (beta / xi) * (((n / nu) * alpha) ** (-xi) - 1)

    return tau, {"u": u, "xi": xi, "beta": beta, "n_tail": nu}

def compute_metrics_with_pot_thresholding(scores):
    # Fit POT
    threshold, info = fit_pot_threshold(scores["train"], q=0.98, alpha=1e-3)

    print("Threshold:", threshold)
    print("GPD params:", info)

    detection_rate = np.mean(scores["actor2_test"] > threshold)
    false_positive_rate = np.mean(scores["actor1_test"] > threshold)

    print(detection_rate, false_positive_rate)
