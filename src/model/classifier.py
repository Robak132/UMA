from utils import split_into_train_test, wrap_labels_with_predictions_to_dataframe, calculate_accuracy


class AbstractClassifier:
    def train(self, x, y):
        raise Exception("This is an abstract method")

    def predict(self, x):
        raise Exception("This is an abstract method")

    def show(self, dataset, train_test_ratio=0.3):
        x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, train_test_ratio)
        self.train(x_train, y_train)
        predictions = self.predict(x_test)
        return wrap_labels_with_predictions_to_dataframe(y_test, predictions)

    def calc_accuracy(self, dataset, iterations=30, train_test_ratio=0.3):
        accuracy = 0
        for i in range(iterations):
            x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, train_test_ratio)
            self.train(x_train, y_train)
            predictions = self.predict(x_test)
            accuracy += calculate_accuracy(y_test, predictions)
        return accuracy/iterations
