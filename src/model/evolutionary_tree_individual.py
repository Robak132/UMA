import numpy as np
from utils import calculate_accuracy


class EvolutionaryTreeIndividual:
    def __init__(self, x, y, division_node_prob=0.3, mutation_change_type_prob=0.2, max_depth=20):
        self.x = x
        self.y = y
        self.mutation_change_type_prob = mutation_change_type_prob
        self.max_depth = max_depth
        self.root = DivisionNode(x, y, division_node_prob, max_depth)
        self.score = 0

    def get_max_depth(self):
        return self.max_depth - self.root.get_max_depth()

    def predict(self, x):
        predictions = []
        for index, row in x.iterrows():
            predictions.append(self.root.proceed(row))
        return predictions

    def evaluate(self, x, y, alpha, beta):
        self.score = alpha * calculate_accuracy(y, self.predict(x)) + beta * self.get_size()

    def get_size(self):
        return len(self.get_nodes())

    def mutate(self):
        node_to_mutate = np.random.choice(self.get_nodes())
        if np.random.rand() > self.mutation_change_type_prob:
            node_to_mutate.mutate()
        else:
            if node_to_mutate is LeafNode:
                self.exchange_node(node_to_mutate, DivisionNode(self.x, self.y, 0, node_to_mutate.depth))
            elif node_to_mutate is DivisionNode:
                self.exchange_node(node_to_mutate, LeafNode(node_to_mutate.y, node_to_mutate.depth))

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


class AbstractNode:
    def proceed(self, record):
        raise Exception("This is an abstract method.")


def create_random_node(x, y, division_node_prob, depth):
    if np.random.rand() < division_node_prob and depth >= 0:
        return DivisionNode(x, y, division_node_prob, depth-1)
    else:
        return LeafNode(y, depth-1)


class DivisionNode(AbstractNode):
    def __init__(self, x, y, division_node_prob, depth):
        self.division_node_prob = division_node_prob
        self.depth = depth
        self.train_data_x = x

        self.left = create_random_node(x, y, division_node_prob, depth)
        self.right = create_random_node(x, y, division_node_prob, depth)

        self.attribute = np.random.choice(x.columns)
        self.value = np.random.uniform(x[self.attribute].min(), x[self.attribute].max())

    def exchange_node(self, node_to_exchange, new_node):
        if self.left == node_to_exchange:
            self.left = new_node
        elif self.right == node_to_exchange:
            self.right = new_node
        else:
            if self.left is DivisionNode:
                self.left.exchange_node(node_to_exchange, new_node)
            if self.right is DivisionNode:
                self.right.exchange_node(node_to_exchange, new_node)

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

    def mutate(self):
        self.attribute = np.random.choice(self.train_data_x.columns)
        self.value = np.random.uniform(self.train_data_x[self.attribute].min(), self.train_data_x[self.attribute].max())

    def __repr__(self):
        return f"Node({self.attribute}:{self.value}, ({self.left.__repr__()}, {self.right.__repr__()})"


class LeafNode(AbstractNode):
    def __init__(self, y, depth):
        self.value = np.random.choice(np.unique(y))
        self.depth = depth
        self.train_data_y = y

    def get_max_depth(self):
        return self.depth

    def proceed(self, record):
        return self.value

    def get_nodes(self):
        return [self]

    def assign_new_depth(self, depth):
        self.depth = depth

    def mutate(self):
        self.value = np.random.choice(np.unique(self.train_data_y))

    def __repr__(self):
        return f"Node({self.value})"
