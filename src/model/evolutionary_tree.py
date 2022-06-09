import threading
from copy import deepcopy

import numpy as np

from model.classifier import AbstractClassifier
from model.tree_individual import TreeIndividual


class EvolutionaryTreeClassifier(AbstractClassifier):
    def __init__(self, alpha=1, beta=-1, max_generations=100, division_node_prob=0.3, max_depth=20, tournament_size=2,
                 elite_size=1):
        self.alpha = alpha
        self.beta = beta
        self.max_depth = max_depth
        self.max_generations = max_generations
        self.division_node_prob = division_node_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        self.best_tree = None

    def train(self, x, y):
        trees = self.initialise(x, y)
        trees = self.score_trees(x, y, trees)
        for generation in range(self.max_generations):
            selected_trees = self.selection(trees)
            mutated_trees = self.mutate_trees(selected_trees)
            trees = self.succession(trees, mutated_trees)
            trees = self.score_trees(x, y, trees)
            print(f"Epoch: {generation} - best accuracy: {trees[0].score}")
        self.best_tree = trees[0]

    def predict(self, x):
        return self.best_tree.predict(x)

    def initialise(self, x, y, population=20):
        trees = []
        for i in range(population):
            tree = TreeIndividual(x, y, division_node_prob=self.division_node_prob, max_depth=self.max_depth)
            trees.append(tree)
        return trees

    def score_trees(self, x, y, trees):
        for tree in trees:
            tree.evaluate(x, y, self.alpha, self.beta)
        return sorted(trees, key=lambda t: t.score, reverse=True)

    def mutate_trees(self, trees):
        for tree in trees:
            tree.mutate()
        return trees

    def succession(self, trees, mutated_trees):
        return trees[:self.elite_size] + mutated_trees[self.elite_size:]

    def selection(self, trees):
        new_trees = []
        for _ in trees:
            opponents = [trees[np.random.randint(len(trees))] for _ in range(self.tournament_size)]
            new_trees.append(deepcopy(max(opponents, key=lambda t: t.score)))
        return new_trees
