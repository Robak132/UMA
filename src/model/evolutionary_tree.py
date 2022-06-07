from model.classifier import AbstractClassifier
from model.evolutionary.tree_individual import TreeIndividual
from utils import calculate_accuracy


class EvolutionaryTreeClassifier(AbstractClassifier):
    def __init__(self, dataset):
        self.dataset = dataset
        self.best_tree = None

    def train(self, x, y):
        trees = self.init_trees(x, y)
        trees = self.score_trees(x, y, trees)
        trees = self.mutate_trees(trees)
        trees = self.succession(trees)

        self.best_tree = trees[0]

    def predict(self, x):
        return self.best_tree.predict(x)

    @staticmethod
    def init_trees(x, y, population=20):
        trees = []
        for i in range(population):
            tree = TreeIndividual(x, y)
            trees.append(tree)
        return trees

    @staticmethod
    def score_trees(x, y, trees):
        for tree in trees:
            tree.evaluate(x, y)
        return sorted(trees, key=lambda x: x.score, reverse=True)

    def mutate_trees(self, trees):
        return trees

    def succession(self, trees):
        return trees
