import random
import threading

from dataset import *
from model.decision_tree import DecisionTreeClassifier
from model.evolutionary_tree import EvolutionaryTreeClassifier
import numpy as np


def test_evolutionary_tree(dataset):
    tree = EvolutionaryTreeClassifier(dataset)
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
    # d1 = load_dataset(", ";")
    # d2 = load_car_evaluation()
    # d3 = load_orders_data()
    # d4 = load_titanic()
    # d5 = load_wine()
    # d6 = load_wine_quality_red()
    # print(d6)
    # sth = Model({"mut_prob": 0.5})
    # # sth.calculate()
    # parameters = {"mut_prob": [0.5, 0.1], "cross_prob": [0.3, 0.2]}
    # calculate_for_all_combinations(parameters)

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
        thread = threading.Thread(target=lambda: test_evolutionary_tree(dataset))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
