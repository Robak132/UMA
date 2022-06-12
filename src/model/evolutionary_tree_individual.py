"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

from enum import Enum
import numpy as np
from utils import calculate_accuracy


class EvolutionaryTreeIndividual:
    def __init__(self, x, y, division_node_prob=0.3, max_depth=20):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.root = DivisionNode(x, y, division_node_prob, max_depth, None)
        self.score = 0

    def predict(self, x):
        predictions = []
        for index, row in x.iterrows():
            predictions.append(self.root.proceed(row))
        return predictions

    def evaluate(self, x, y, alpha, beta):
        self.score = alpha * calculate_accuracy(y.tolist(), self.predict(x)) + beta * self.get_size()

    def get_size(self):
        return len(self.get_nodes())

    def mutate(self, mutation_change_type_prob):
        node_to_mutate = np.random.choice(self.get_nodes(with_root=False))
        if np.random.rand() > mutation_change_type_prob:
            # Changing parameters
            node_to_mutate.mutate_division()
        else:
            # Change tree structure
            node_to_mutate.change_node_type()

    def exchange_node(self, node_to_exchange, new_node):
        if self.root == node_to_exchange:
            self.root = new_node
        else:
            self.root.exchange_node(node_to_exchange, new_node)

    def get_nodes(self, with_root=True):
        if with_root:
            return self.root.get_nodes()
        else:
            return [*self.root.right.get_nodes(), *self.root.left.get_nodes()]

    def get_random_node(self, with_root=True):
        if with_root:
            return np.random.choice(self.get_nodes())
        else:
            return np.random.choice(self.get_nodes(with_root=False))

    def __repr__(self):
        return f"Tree(score = {self.score})"


class NodeType(Enum):
    LEAF_NODE = 0,
    DIVISION_NODE = 1


class GenericNode:
    def __init__(self, x, y, depth, parent):
        self.x = x
        self.y = y
        self.depth = depth
        self.parent = parent
        self.type = None

    def proceed(self, record):
        raise Exception("This is an abstract method.")

    def mutate_division(self):
        raise Exception("This is an abstract method.")

    def replace_node(self, new_node):
        if self.parent is None:
            return

        new_node.parent = self.parent

        if self.parent.left == self:
            self.parent.left = new_node
        else:
            self.parent.right = new_node


class DivisionNode(GenericNode):
    def __init__(self, x, y, division_node_prob, depth, parent):
        super().__init__(x, y, depth, parent)

        self.left = self.create_random_node(x, y, division_node_prob, depth, self)
        self.right = self.create_random_node(x, y, division_node_prob, depth, self)

        self.type = NodeType.DIVISION_NODE
        self.division_node_prob = division_node_prob
        self.attribute = np.random.choice(x.columns)
        self.value = np.random.uniform(x[self.attribute].min(), x[self.attribute].max())

    @staticmethod
    def create_random_node(x, y, division_node_prob, depth, parent):
        if np.random.rand() < division_node_prob and depth > 1:
            return DivisionNode(x, y, division_node_prob, depth - 1, parent)
        else:
            return LeafNode(x, y, depth - 1, parent)

    def change_node_type(self):
        self.replace_node(LeafNode(self.x, self.y, self.depth, self.parent))

    def assign_new_depth(self, new_depth):
        self.depth = new_depth
        self.right.assign_new_depth(new_depth-1)
        self.left.assign_new_depth(new_depth-1)

    def get_nodes(self):
        return [self, *self.right.get_nodes(), *self.left.get_nodes()]

    def get_max_depth(self):
        return min(self.left.get_max_depth(), self.right.get_max_depth())

    def proceed(self, record):
        if record[self.attribute] > self.value:
            return self.left.proceed(record)
        else:
            return self.right.proceed(record)

    def mutate_division(self):
        self.attribute = np.random.choice(self.x.columns)
        self.value = np.random.uniform(self.x[self.attribute].min(), self.x[self.attribute].max())

    def __repr__(self):
        return f"Node({self.attribute}:{self.value}, ({self.left.__repr__()}, {self.right.__repr__()})"


class LeafNode(GenericNode):
    def __init__(self, x, y, depth, parent):
        super().__init__(x, y, depth, parent)

        self.type = NodeType.LEAF_NODE
        self.value = np.random.choice(np.unique(y))

    def replace_node(self, new_node):
        if self.parent is None:
            return

        new_node.parent = self.parent
        if self.parent.left == self:
            self.parent.left = new_node
        else:
            self.parent.right = new_node

    def change_node_type(self):
        self.replace_node(DivisionNode(self.x, self.y, 0, self.depth, self.parent))

    def get_max_depth(self):
        return self.depth

    def proceed(self, record):
        return self.value

    def get_nodes(self):
        return [self]

    def assign_new_depth(self, depth):
        self.depth = depth

    def mutate_division(self):
        self.value = np.random.choice(np.unique(self.y))

    def __repr__(self):
        return f"Node({self.value})"
