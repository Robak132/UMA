import random
import numpy as np
import pandas as pd

from dataset import Dataset, BreastTissueDataset
from model.decision_tree_classifier import DecisionTreeClassifier
from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier


def test_decision_tree():
    random.seed(2137)
    np.random.seed(2137)

    xorDataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    tree = DecisionTreeClassifier()
    tree.train(xorDataset.x, xorDataset.y)
    predictions = tree.predict(pd.DataFrame([[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert predictions == [0, 1, 0, 1]


def test_evolutionary_tree():
    random.seed(2137)
    np.random.seed(2137)

    xorDataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    tree = EvolutionaryTreeClassifier(alpha=1,
                                      beta=-0.1,
                                      max_generations=100,
                                      division_node_prob=0.3,
                                      max_depth=20,
                                      tournament_size=2,
                                      elite_size=1)

    tree.train(xorDataset.x, xorDataset.y)
    predictions = tree.predict(pd.DataFrame([[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert predictions == [0, 1, 0, 1]
