import random
import threading

from dataset import *
from model.decision_tree_classifier import DecisionTreeClassifier
from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier
import numpy as np


def test_evolutionary_tree(dataset, config):
    print(f"{dataset.name} [evolution] started\n", end="")
    tree = EvolutionaryTreeClassifier(config)
    # tree = EvolutionaryTreeClassifier(alpha=100, beta=-1, division_node_prob=0.3, max_depth=50)
    stats = tree.experiment(dataset, f"../output/{dataset.name}/evolutionary/stats.txt", iterations=5)
    print(f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


def test_classic_tree(dataset):
    print(f"{dataset.name} [standard] started\n", end="")
    normal_tree = DecisionTreeClassifier()
    stats = normal_tree.experiment(dataset, f"../output/{dataset.name}/classic/stats.txt", iterations=5)
    print(f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)

    datasets = [
        BreastTissueDataset("../data/extracted/breast_tissue.csv"),
        CarEvaluationDataset("../data/extracted/car_evaluation.csv"),
        # TitanicDataset("../data/extracted/titanic.csv"),
        # WineDataset("../data/extracted/wine.csv"),
        # RedWineQualityDataset("../data/extracted/winequality_red.csv"),
    ]

    evolutionary_config = {
        "alpha": 100,
        "beta": -0.1,
        "max_depth": 50,
        "max_generations": 1000,
        "population_size": 50,
        "crossover": True,
        "division_node_prob": 0.5,
        "mutation_change_type_prob": 0.2,
        "tournament_size": 2,
        "elite_size": 1
    }
    for dataset in datasets:
        test_classic_tree(dataset)
        test_evolutionary_tree(dataset, evolutionary_config)
