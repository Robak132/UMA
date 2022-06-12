"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

from sklearn import tree
from utils import calculate_accuracy

from model.generic_classifier import GenericClassifier


class DecisionTreeClassifier(GenericClassifier):
    def __init__(self):
        super().__init__()
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.logger = []
        self.clf.fit(x, y)
        self.logger.append({
            "size": self.clf.tree_.node_count,
            "depth": self.clf.get_depth(),
            "accuracy_on_test": calculate_accuracy(y.tolist(), self.predict(x))
        })

    def predict(self, x) -> list:
        return self.clf.predict(x).tolist()
