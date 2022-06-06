from sklearn import tree


class DecisionTreeClassifier:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)  # TODO sprawdzić czy to się tak ładnie zwróci po kolei
