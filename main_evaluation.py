import yaml

from src.utils.seed import set_seed
from src.evaluation.compute_errors import compute_errors
from src.evaluation.compute_scores import compute_scores, evalutation_pipeline
from src.evaluation.compute_metrics import compute_metrics

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    compute_errors(config)
    # scores = compute_scores(config)
    # evalutation_pipeline(config)
    # print("train mean and std: ", scores["train"].mean(), scores["train"].std())
    # print("val mean and std: ", scores["val"].mean(), scores["val"].std())
    # print("actor 2 test mean and std: ", scores["actor2_test"].mean(), scores["actor2_test"].std())
    # print("actor 1 test mean and std: ", scores["actor1_test"].mean(), scores["actor1_test"].std())
    # compute_metrics(scores, config)


    # compute_metrics_with_pot_thresholding(scores)

if __name__ == "__main__":
    main_evaluation()

