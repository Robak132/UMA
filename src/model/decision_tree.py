from sklearn import tree

from utils import wrap_labels_with_predictions_to_dataframe, calculate_accuracy, split_into_train_test


class DecisionTreeClassifier:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)
        # TODO sprawdzić czy to się tak ładnie zwróci po kolei

    def show(self, dataset):
        x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, 0.3)
        self.train(x_train, y_train)
        predictions = self.predict(x_test)
        return wrap_labels_with_predictions_to_dataframe(y_test, predictions)

    def calc_accuracy(self, dataset, iterations=30):
        accuracy = 0
        for i in range(iterations):
            x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, 0.3)
            self.train(x_train, y_train)
            predictions = self.predict(x_test)
            accuracy += calculate_accuracy(y_test, predictions)
        return accuracy/iterations
