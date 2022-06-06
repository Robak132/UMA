from loaders import *
from model.evolutionary.tree_individual import TreeIndividual
# from model.evolutionary_tree import EvolutionaryTreeClassifier
import numpy as np

import sys
sys.setrecursionlimit(10000)

if __name__ == "__main__":
    X, Y = load_breast_tissue()
    tree = TreeIndividual(X, Y)
    tree.train(X, Y)
    print(tree.predict(X))
    # attr = np.random.choice(X.columns)
    # print(X[attr])
    # print(X[attr].min())
    # print(X[attr].max())

    # print(np.random.uniform())
