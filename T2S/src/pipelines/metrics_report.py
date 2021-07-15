import pandas as pd
from pathlib import Path
import os

ROOT_DIR = os.path.join(Path(__file__).parent.parent.parent)
results_path = os.path.join(ROOT_DIR, "resultfiles", "triple_evaluation")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

if __name__ == '__main__':
    metrics = pd.read_json(os.path.join(results_path, "metrics_by_topic.json"))
    metrics_random = pd.read_json(os.path.join(results_path, "random_evaluation.json"))
    metrics_stanford = pd.read_json(os.path.join(results_path, "stanford_evaluation.json"))

    metrics = metrics[["leven", "topic"]]
    metrics_random = metrics_random[["leven", "topic"]]
    metrics_stanford = metrics_stanford[["leven", "topic"]]
    print(metrics)
    print()

    temp = pd.merge(metrics, metrics_random, on="topic", suffixes=["_t2s", "_random"])
    temp = pd.merge(temp, metrics_stanford, on="topic")
    temp.rename(columns={"leven": "leven_stanford"}, inplace=True)
    temp = temp[["topic", "leven_t2s", "leven_random", "leven_stanford"]]
    print(temp)

    temp.to_csv(os.path.join(results_path, "metrics_comparison.csv"), index=False)
