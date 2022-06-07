import os
from pathlib import Path

import numpy as np

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

    def save_stats(self, dataset, file, iterations=30, train_test_ratio=0.3):
        accuracy = []
        for i in range(iterations):
            x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, train_test_ratio)
            self.train(x_train, y_train)
            predictions = self.predict(x_test)
            current_accuracy = calculate_accuracy(y_test, predictions)
            print(f"Initialisation {i+1}/{iterations}: Accuracy: {current_accuracy}")
            accuracy.append(current_accuracy)

        stats = (np.mean(accuracy), np.std(accuracy), np.min(accuracy), np.max(accuracy))
        self._save_file(file, stats)
        return stats

    @staticmethod
    def _save_file(file: str, stats: tuple):
        os.makedirs(os.path.split(file)[0], exist_ok=True)
        with open(file, "w+") as file:
            file.write(f"mean: {stats[0]}\n")
            file.write(f"standard_deviation: {stats[1]}\n")
            file.write(f"min: {stats[2]}\n")
            file.write(f"max: {stats[3]}\n")
