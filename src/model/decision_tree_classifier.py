from sklearn import tree

from model.abstract_classifier import AbstractClassifier


class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)
