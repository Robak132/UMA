from sklearn import tree

from utils import wrap_labels_with_predictions_to_dataframe, calculate_accuracy


class DecisionTreeClassifier:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, dataset):
        self.clf.fit(dataset.x_train, dataset.y_train)

    def predict(self, dataset):
        return self.clf.predict(dataset.x_test)  # TODO sprawdzić czy to się tak ładnie zwróci po kolei

    def calc_accuracy(self, dataset):
        predictions = self.predict(dataset)
        labels = dataset.y_test

        outcome = wrap_labels_with_predictions_to_dataframe(labels, predictions)
        accuracy = calculate_accuracy(predictions, labels)

        return accuracy, outcome
