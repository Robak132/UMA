"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

import numpy as np
import pandas as pd


def get_dictionary_combinations(parameter_list):
    current_key = list(parameter_list.keys())[0]
    if len(parameter_list) == 1:
        return [[{current_key: parameter_value}] for parameter_value in parameter_list[current_key]]
    else:
        popped_dictionary = parameter_list.copy()
        popped_dictionary.pop(current_key)
        return [[{current_key: parameter_value}, *some_list] for parameter_value in parameter_list[current_key] for
                some_list in get_dictionary_combinations(popped_dictionary.copy())]


def merge_list_of_dicts_into_one_dict(lists):
    merged_dict = {}
    for dictionary in lists:
        for key in dictionary.keys():
            merged_dict[key] = dictionary[key]
    return merged_dict


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
    return np.sum(np.array(predictions) == np.array(labels)) / len(labels)


def format_dict_to_str(dictionary: dict) -> str:
    result_string = ""
    for element in dictionary:
        result_string += str(element) + "_" + str(dictionary[element]) + "_"
    return result_string[:-1]


def join_trees(node, sub_tree):
    sub_tree.assign_new_depth(node.depth)
    node.replace_node(sub_tree)
