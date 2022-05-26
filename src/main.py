from loaders import *
from model.model import Model


def calculate_for_all_combinations(parameters):
    combinations = get_dictionary_combinations(parameters)
    for params in combinations:
        model = Model(params)
        model.calculate()


def get_dictionary_combinations(parameter_list):
    current_key = list(parameter_list.keys())[0]
    if len(parameter_list) == 1:
        return [[{current_key:parameter_value}] for parameter_value in parameter_list[current_key]]
    else:
        popped_dictionary = parameter_list.copy()
        popped_dictionary.pop(current_key)
        return [[{current_key: parameter_value}, *some_list] for parameter_value in parameter_list[current_key] for some_list in get_dictionary_combinations(popped_dictionary.copy())]


def get_combinations(parameter_list):
    if len(parameter_list) == 1:
        return [[parameter_value] for parameter_value in parameter_list[0]]
    else:
        return [[parameter_value, *some_list] for parameter_value in parameter_list[0] for some_list in get_combinations(parameter_list[1:])]


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

if __name__ == "__main__":
    # d1 = load_dataset("../data/extracted/breast_tissue.csv", ";")
    d1 = load_breast_tissue()
    d2 = load_car_evaluation()
    d3 = load_orders_data()
    d4 = load_titanic()
    d5 = load_wine()
    d6 = load_wine_quality_red()
    # print(d6)
    sth = Model({"mut_prob": 0.5})
    # sth.calculate()
    parameters = {"mut_prob": [0.5, 0.1], "cross_prob": [0.3, 0.2]}
    calculate_for_all_combinations(parameters)

