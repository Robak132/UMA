import random

from dataset import BreastTissueDataset
from loaders import *
from model.evolutionary.tree_individual import TreeIndividual
# from model.evolutionary_tree import EvolutionaryTreeClassifier
import numpy as np


if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)

    dataset = BreastTissueDataset("../data/extracted/breast_tissue.csv")

    trees.sort(key=lambda x: x.score)
    pass
    # attr = np.random.choice(X.columns)
    # print(X[attr])
    # print(X[attr].min())
    # print(X[attr].max())

    # print(np.random.uniform())
