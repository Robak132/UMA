"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

import random
import numpy as np
import pandas as pd

from dataset import Dataset
from model.decision_tree_classifier import DecisionTreeClassifier
from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier


def test_decision_tree():
    random.seed(2137)
    np.random.seed(2137)

    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    tree = DecisionTreeClassifier()
    tree.train(xor_dataset.x, xor_dataset.y)
    predictions = tree.predict(pd.DataFrame([[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert predictions == [0, 1, 0, 1]


def test_evolutionary_tree():
    random.seed(2137)
    np.random.seed(2137)

    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    evolutionary_config = {
        "alpha": 1,
        "beta": -0.01,
        "max_depth": 50,
        "max_generations": 100,
        "population_size": 50,
        "crossover": True,
        "division_node_prob": 0.3,
        "mutation_change_type_prob": 0.2,
        "tournament_size": 2,
        "elite_size": 1
    }
    tree = EvolutionaryTreeClassifier(evolutionary_config)

    tree.train(xor_dataset.x, xor_dataset.y)
    predictions = tree.predict(pd.DataFrame([[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert predictions == [0, 1, 0, 1]
