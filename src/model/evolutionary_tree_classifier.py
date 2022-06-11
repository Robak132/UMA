from copy import deepcopy

import numpy as np

from model.abstract_classifier import AbstractClassifier
from model.evolutionary_tree_individual import EvolutionaryTreeIndividual


class EvolutionaryTreeClassifier(AbstractClassifier):
    def __init__(self, config):
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.max_depth = config['max_depth']
        self.max_generations = config['max_generations']
        self.division_node_prob = config['division_node_prob']
        self.mutation_change_type_prob = config['mutation_change_type_prob']
        self.tournament_size = config['tournament_size']
        self.elite_size = config['elite_size']
        self.crossover = config['crossover']
        self.population_size = config['population_size']

        self.best_tree = None

    def train(self, x, y):
        trees = self.initialise(x, y, self.population_size)
        trees = self.score_trees(x, y, trees)
        for generation in range(self.max_generations):
            selected_trees = self.selection(trees)
            if self.crossover:
                crossovered_trees = self.crossover_trees(selected_trees)
                mutated_trees = self.mutate_trees(crossovered_trees)
            else:
                mutated_trees = self.mutate_trees(selected_trees)
            trees = self.succession(trees, mutated_trees)
            trees = self.score_trees(x, y, trees)
            print(f"Epoch: {generation} - best tree score: {trees[0].score}")
        self.best_tree = trees[0]

    def predict(self, x) -> list:
        return self.best_tree.predict(x)

    def initialise(self, x, y, population):
        trees = []
        for i in range(population):
            tree = EvolutionaryTreeIndividual(x, y, self.division_node_prob, self.max_depth)
            trees.append(tree)
        return trees

    def score_trees(self, x, y, trees):
        for tree in trees:
            tree.evaluate(x, y, self.alpha, self.beta)
        return sorted(trees, key=lambda t: t.score, reverse=True)

    def mutate_trees(self, trees):
        for tree in trees:
            tree.mutate(self.mutation_change_type_prob)
        return trees

    @staticmethod
    def crossover_trees(trees):
        for tree in trees:
            node = tree.get_random_node(with_root=False)
            new_subtree = deepcopy(np.random.choice(trees).get_random_node())
            new_subtree.assign_new_depth(node.depth)
            node.replace_node(new_subtree)
        return trees

    def succession(self, trees, mutated_trees):
        return trees[:self.elite_size] + mutated_trees[self.elite_size:]

    def selection(self, trees):
        new_trees = []
        for _ in trees:
            opponents = [trees[np.random.randint(len(trees))] for _ in range(self.tournament_size)]
            new_trees.append(deepcopy(max(opponents, key=lambda t: t.score)))
        return new_trees
