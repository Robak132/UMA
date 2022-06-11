from sklearn import tree
from utils import calculate_accuracy

from model.abstract_classifier import AbstractClassifier


class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.logger = []
        self.clf.fit(x, y)
        self.logger.append({
            "size": self.clf.tree_.node_count,
            "depth": self.clf.get_depth(),
            "accuracy_on_test": calculate_accuracy(y.tolist(), self.predict(x))
        })

    def get_logger(self) -> list:
        return self.logger

    def predict(self, x) -> list:
        return self.clf.predict(x).tolist()
