import random

from dataset import *
from loaders import *
from model.decision_tree import DecisionTreeClassifier
from model.evolutionary_tree import EvolutionaryTreeClassifier
from utils import split_into_train_test, wrap_labels_with_predictions_to_dataframe, calculate_accuracy
from utils import calculate_accuracy_from_dataframe
import pandas as pd
import numpy as np

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

    print("Standard Decision Tree")
    for dataset in datasets:
        normal_tree = DecisionTreeClassifier()
        accuracy = normal_tree.calc_accuracy(dataset)
        print(f"{dataset.name} accuracy: {accuracy}")

    print()
    print("Evolutionary Decision Tree")
    for dataset in datasets:
        tree = EvolutionaryTreeClassifier(dataset)
        accuracy = tree.calc_accuracy(dataset)
        print(f"{dataset.name} accuracy: {accuracy}")
