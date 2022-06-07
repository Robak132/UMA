from model.classifier import AbstractClassifier
from model.evolutionary.tree_individual import TreeIndividual


class EvolutionaryTreeClassifier(AbstractClassifier):
    def __init__(self, alpha=1, beta=-1, division_node_prob=0.3, max_depth=20):
        self.alpha = alpha
        self.beta = beta
        self.max_depth = max_depth
        self.division_node_prob = division_node_prob

        self.best_tree = None

    def train(self, x, y):
        trees = self.init_trees(x, y)
        trees = self.score_trees(x, y, trees)
        trees = self.mutate_trees(trees)
        trees = self.succession(trees)

        self.best_tree = trees[0]

    def predict(self, x):
        return self.best_tree.predict(x)

    def init_trees(self, x, y, population=20):
        trees = []
        for i in range(population):
            tree = TreeIndividual(x, y, division_node_prob=self.division_node_prob, max_depth=self.max_depth)
            trees.append(tree)
        return trees

    def score_trees(self, x, y, trees):
        for tree in trees:
            tree.evaluate(x, y, self.alpha, self.beta)
        return sorted(trees, key=lambda x: x.score, reverse=True)

    def mutate_trees(self, trees):
        return trees

    def succession(self, trees):
        return trees
