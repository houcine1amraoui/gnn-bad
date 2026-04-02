import matplotlib.pyplot as plt
import os
import numpy as np

def plot_bins(scores, config):
    model_name = config["evaluation"]["model"]

    eval_results_folder = config["evaluation"]["eval_results_folder"]

    eval_results_per_model_folder = f"{eval_results_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(eval_results_per_model_folder, exist_ok=True)

    # plots folder
    plots_folder = f"{eval_results_per_model_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)

    plt.hist(scores["train"], bins=100, alpha=0.5, label="actor1_w1")
    plt.hist(scores["actor2_test"], bins=100, alpha=0.5, label="actor2")
    plt.hist(scores["actor1_test"], bins=100, alpha=0.5, label="actor1_w2")
    plt.legend()
    plt.show()
    


def plot_anomaly_scores_distribution(scores, config):
    model_name = config["evaluation"]["model"]

    eval_results_folder = config["evaluation"]["eval_results_folder"]

    eval_results_per_model_folder = f"{eval_results_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(eval_results_per_model_folder, exist_ok=True)

    # plots folder
    plots_folder = f"{eval_results_per_model_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    
    # --- Threshold ---
    threshold_percentile = config["evaluation"]["threshold_percentile"]
    threshold = np.percentile(scores["train"], threshold_percentile)
    print("threshold", threshold)
    threshold = 0.5
    plt.figure(figsize=(15,5))

    # lengths
    n_train = len(scores["train"])
    n_val = len(scores["val"])
    n_actor2 = len(scores["actor2_test"])
    n_actor1 = len(scores["actor1_test"])

    # x ranges (shifted)
    x_train = range(0, n_train)
    x_val = range(n_train, n_train + n_val)
    x_actor2 = range(n_train + n_val, n_train + n_val + n_actor2)
    x_actor1 = range(n_train + n_val + n_actor2, n_train + n_val + n_actor2 + n_actor1)

    # plot
    plt.plot(x_train, scores["train"], label="Actor 1 (Train)")
    plt.plot(x_val, scores["val"], label="Validation")
    plt.plot(x_actor2, scores["actor2_test"], label="Actor 2 (Test)")
    plt.plot(x_actor1, scores["actor1_test"], label="Actor 1 (Test)")

    # optional: vertical separators
    plt.axvline(n_train, linestyle="--")
    plt.axvline(n_train + n_val, linestyle="--")
    plt.axvline(n_train + n_val + n_actor2, linestyle="--")

    plt.axhline(y=threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.legend()
    plt.title("Anomaly Scores (Concatenated Timeline)")

    plt.savefig(f"{plots_folder}/anomlay_scores_distribution.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_boxplot(scores, eval_results_folder):

    plt.figure(figsize=(8, 5))

    data = [
        scores["train"],
        scores["val"],
        scores["actor2_test"],
        scores["actor1_test"]
    ]

    plt.boxplot(data,
                showfliers=False)

    plt.title("Score Distribution Comparison")
    plt.ylabel("Score")

    #
    os.makedirs(f"{eval_results_folder}/plots", exist_ok=True)
    plt.savefig(f"{eval_results_folder}/plots/boxplot.png", dpi=300, bbox_inches="tight")

    plt.show()


