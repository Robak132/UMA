from loaders import *
from model.decision_tree import DecisionTreeClassifier
from utils import split_into_train_test, wrap_labels_with_predictions_to_dataframe, calculate_accuracy, calculate_accuracy_from_dataframe
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # d1 = load_dataset("../data/extracted/breast_tissue.csv", ";")
    X, Y = load_breast_tissue()
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

    X_train, X_test, Y_train, Y_test = split_into_train_test(X, Y, 0.3)

    normal_tree = DecisionTreeClassifier()
    normal_tree.train(X_train, Y_train)
    predictions = normal_tree.predict(X_test)

    labels = Y_test
    outcome = wrap_labels_with_predictions_to_dataframe(labels, predictions)
    print(outcome)

    outcome.to_csv('breast_tissue.csv')
    print(pd.read_csv('breast_tissue.csv', index_col=0))
    print(calculate_accuracy(predictions, labels))
    print(calculate_accuracy_from_dataframe(pd.read_csv('breast_tissue.csv', index_col=0)))



