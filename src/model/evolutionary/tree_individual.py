import numpy as np
from utils import calculate_accuracy


class TreeIndividual:
    def __init__(self, X, Y):
        self.root = DivisionNode(X, Y)

    def train(self, X, Y):
        predictions = self.predict(X)
        print(calculate_accuracy(Y, predictions))

    def predict(self, X):
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.predict_one(row))
        return predictions

    def predict_one(self, record):
        return self.root.proceed(record)


def create_random_node(X, Y):
    if np.random.rand() < 0.3:
        return DivisionNode(X, Y)
    else:
        return LeafNode(Y)


class DivisionNode:
    def __init__(self, X, Y):
        self.left = create_random_node(X, Y)
        self.right = create_random_node(X, Y)
        self.division = Division(X)

    def proceed(self, record):
        if self.division.proceed(record):
            return self.left.proceed(record)
        else:
            return self.right.proceed(record)


class LeafNode:
    def __init__(self, Y):
        self.value = np.random.choice(Y)

    def proceed(self, record):
        return self.value


class Division:
    def __init__(self, X):
        self.attribute = np.random.choice(X.columns)
        self.value = np.random.uniform(X[self.attribute].min(), X[self.attribute].max())

    def proceed(self, record):
        if record[self.attribute] > self.value:
            return True
        else:
            return False
