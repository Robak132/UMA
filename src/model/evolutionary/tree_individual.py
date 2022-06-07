import numpy as np
from utils import calculate_accuracy


class TreeIndividual:
    def __init__(self, x, y, division_node_prob=0.3, max_depth=20):
        self.root = DivisionNode(x, y, division_node_prob, max_depth)
        self.score = 0

    def predict(self, x):
        predictions = []
        for index, row in x.iterrows():
            predictions.append(self.root.proceed(row))
        return predictions

    def evaluate(self, x, y):
        self.score = calculate_accuracy(y, self.predict(x))

    def __repr__(self):
        return f"Tree(score = {self.score})"


class AbstractNode:
    def proceed(self, record):
        raise Exception("This is an abstract method.")


class DivisionNode(AbstractNode):
    def __init__(self, x, y, division_node_prob, depth):
        self.division_node_prob = division_node_prob
        self.depth = depth

        self.left = self.create_random_node(x, y)
        self.right = self.create_random_node(x, y)

        self.attribute = np.random.choice(x.columns)
        self.value = np.random.uniform(x[self.attribute].min(), x[self.attribute].max())

    def proceed(self, record):
        if record[self.attribute] > self.value:
            return self.left.proceed(record)
        else:
            return self.right.proceed(record)

    def create_random_node(self, x, y):
        if np.random.rand() < self.division_node_prob and self.depth >= 0:
            return DivisionNode(x, y, self.division_node_prob, self.depth-1)
        else:
            return LeafNode(y, self.depth-1)

    def __repr__(self):
        return f"Node({self.value}, ({self.left.__repr__()}, {self.right.__repr__()})"


class LeafNode(AbstractNode):
    def __init__(self, y, depth):
        self.value = np.random.choice(y)
        self.depth = depth

    def proceed(self, record):
        return self.value

    def __repr__(self):
        return f"Node({self.value})"
