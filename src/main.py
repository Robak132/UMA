import random
import threading

from dataset import *
from model.decision_tree import DecisionTreeClassifier
from model.evolutionary_tree import EvolutionaryTreeClassifier
import numpy as np


def test_evolutionary_tree(dataset):
    tree = EvolutionaryTreeClassifier(alpha=100, beta=-1, division_node_prob=0.3, max_depth=20)
    stats = tree.save_stats(dataset, f"../output/{dataset.name}/evolutionary/stats.txt")
    print(f"{dataset.name} [evolution] finished\n"
          f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


def test_classic_tree():
    normal_tree = DecisionTreeClassifier()
    stats = normal_tree.save_stats(dataset, f"../output/{dataset.name}/classic/stats.txt")
    print(f"{dataset.name} [standard] finished\n"
          f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)

    datasets = [
        BreastTissueDataset("../data/extracted/breast_tissue.csv"),
        CarEvaluationDataset("../data/extracted/car_evaluation.csv"),
        TitanicDataset("../data/extracted/titanic.csv"),
        WineDataset("../data/extracted/wine.csv"),
        RedWineQualityDataset("../data/extracted/winequality_red.csv"),
    ]

    threads = []
    for dataset in datasets:
        test_classic_tree()
        # test_evolutionary_tree(dataset)
        thread = threading.Thread(target=lambda: test_evolutionary_tree(dataset))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
