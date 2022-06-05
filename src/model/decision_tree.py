from sklearn import tree


class DecisionTreeClassifier:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X) #TODO sprawdzić czy to się tak ładnie zwróci po kolei
