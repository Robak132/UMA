import os

import numpy as np
import pandas as pd


# def calculate_for_all_combinations(parameters):
#     combinations = get_dictionary_combinations(parameters)
#     for params in combinations:
#         model = Model(params)
#         model.calculate()


def get_dictionary_combinations(parameter_list):
    current_key = list(parameter_list.keys())[0]
    if len(parameter_list) == 1:
        return [[{current_key: parameter_value}] for parameter_value in parameter_list[current_key]]
    else:
        popped_dictionary = parameter_list.copy()
        popped_dictionary.pop(current_key)
        return [[{current_key: parameter_value}, *some_list] for parameter_value in parameter_list[current_key] for
                some_list in get_dictionary_combinations(popped_dictionary.copy())]


def get_combinations(parameter_list):
    if len(parameter_list) == 1:
        return [[parameter_value] for parameter_value in parameter_list[0]]
    else:
        return [[parameter_value, *some_list] for parameter_value in parameter_list[0] for some_list in
                get_combinations(parameter_list[1:])]


def get_all_combinations(parameters_lists):
    combinations = []
    for one_list in parameters_lists:
        for number in one_list:
            combinations.append([number])

    for number in parameters_lists[0]:
        for number2 in parameters_lists[1]:
            for number3 in parameters_lists[2]:
                combinations.append([number, number2, number3])
    return combinations


def split_into_train_test(x, y, test_ratio):
    train_indices = np.random.rand(len(x)) > test_ratio
    x_train = x.loc[train_indices]
    x_test = x.loc[~train_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[~train_indices]
    return x_train, x_test, y_train, y_test


def wrap_labels_with_predictions_to_dataframe(labels, predictions):
    outcome = pd.concat([labels, pd.DataFrame(predictions, index=labels.index)], axis=1, ignore_index=True)
    outcome.columns = ["label", "prediction"]
    return outcome


def calculate_accuracy(labels, predictions):
    return np.sum(predictions == labels) / len(labels)


def calculate_accuracy_from_dataframe(outcome_dataframe):
    return np.sum(outcome_dataframe['prediction'] == outcome_dataframe['label']) / len(outcome_dataframe)
