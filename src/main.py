import random

from dataset import BreastTissueDataset
from loaders import *
from model.decision_tree import DecisionTreeClassifier
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

    breastTissueDataset = BreastTissueDataset("../data/extracted/breast_tissue.csv")

    normal_tree = DecisionTreeClassifier()
    normal_tree.train(breastTissueDataset)
    accuracy, result = normal_tree.calc_accuracy(breastTissueDataset)
    print(accuracy)
