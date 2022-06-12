"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

import random
import threading
import time
from datetime import datetime
from dataset import BreastTissueDataset, CarEvaluationDataset, TitanicDataset, WineDataset, RedWineQualityDataset
from model.decision_tree_classifier import DecisionTreeClassifier
from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier

from utils import get_dictionary_combinations, merge_list_of_dicts_into_one_dict
import numpy as np


def test_evolutionary_tree(dataset, evolutionary_config, meta_config, experiment_id):
    print(f"{dataset.name} [evolution] started\n", end="")
    tree = EvolutionaryTreeClassifier(evolutionary_config)
    path = f"../output/{experiment_id}/{dataset.name}/evolutionary"
    tree._save_config(path, evolutionary_config)
    stats = tree.experiment(dataset,
                            path,
                            iterations=meta_config["iterations"],
                            train_test_ratio=meta_config["train_test_ratio"])
    print(f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


def test_classic_tree(dataset, meta_config, experiment_id):
    print(f"{dataset.name} [standard] started\n", end="")
    path = f"../output/{experiment_id}/{dataset.name}/classic"
    normal_tree = DecisionTreeClassifier()
    stats = normal_tree.experiment(dataset,
                                   path,
                                   iterations=meta_config["iterations"],
                                   train_test_ratio=meta_config["train_test_ratio"])
    print(f"mean: {stats[0]}\n"
          f"standard_deviation: {stats[1]}\n"
          f"min: {stats[2]}\n"
          f"max: {stats[3]}\n\n", end="", flush=True)


def run_thread(datasets, evolutionary_config, meta_config, experiment_id):
    for dataset in datasets:
        test_classic_tree(dataset, meta_config, experiment_id)
        test_evolutionary_tree(dataset, evolutionary_config, meta_config, experiment_id)


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

    meta_config = {
        "train_test_ratio": 0.3,
        "iterations": 25
    }
    configs = {
        "alpha": [100],
        "beta": [-0.1],
        "max_depth": [50],
        "max_generations": [250],
        "population_size": [20],
        "crossover": [True, False],
        "division_node_prob": [0.1, 0.3, 0.5],
        "mutation_change_type_prob": [0.5],
        "tournament_size": [2],
        "elite_size": [1]
    }
    threads = []

    for config in get_dictionary_combinations(configs):
        evolutionary_config = merge_list_of_dicts_into_one_dict(config)
        experiment_id = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
        current_thread = threading.Thread(target=lambda: run_thread(datasets, evolutionary_config, meta_config, experiment_id))
        threads.append(current_thread)
        current_thread.start()
        time.sleep(2)

    for thread in threads:
        thread.join()
