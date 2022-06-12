"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

import json
import os
import numpy as np
import pandas as pd

from utils import split_into_train_test, wrap_labels_with_predictions_to_dataframe, calculate_accuracy


class GenericClassifier:
    def __init__(self):
        self.logger = []

    def train(self, x, y):
        raise Exception("This is an abstract method")

    def predict(self, x) -> list:
        raise Exception("This is an abstract method")

    def experiment(self, dataset, path, iterations=30, train_test_ratio=0.3):
        accuracy = []
        for i in range(iterations):
            x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, train_test_ratio)
            self.train(x_train, y_train)
            predictions = self.predict(x_test)
            current_accuracy = calculate_accuracy(y_test, predictions)
            print(f"Initialisation {i+1}/{iterations}: Accuracy: {current_accuracy}")
            accuracy.append(current_accuracy)
            self._save_predictions(path, i, y_test, predictions)
            self._save_logger(path, i, self.logger)

        stats = (np.mean(accuracy), np.std(accuracy), np.min(accuracy), np.max(accuracy))
        self._save_file(path + "/stats.txt", stats)
        return stats

    @staticmethod
    def _save_predictions(path, iteration, labels, predictions):
        os.makedirs(path + "/" + str(iteration), exist_ok=True)
        outcome = wrap_labels_with_predictions_to_dataframe(labels, predictions)
        outcome.to_csv(path + "/" + str(iteration) + "/predictions.csv")

    @staticmethod
    def _save_logger(path, iteration, logger):
        os.makedirs(path + "/" + str(iteration), exist_ok=True)
        logger_dataframe = pd.DataFrame(logger)
        logger_dataframe.to_csv(path + "/" + str(iteration) + "/logger.csv")

    @staticmethod
    def _save_config(path, config):
        os.makedirs(path, exist_ok=True)
        with open(path + "/config.json", "w+") as file:
            json.dump(config, file)

    @staticmethod
    def _save_file(file: str, stats: tuple):
        os.makedirs(os.path.split(file)[0], exist_ok=True)
        with open(file, "w+") as file:
            file.write(f"mean: {stats[0]}\n")
            file.write(f"standard_deviation: {stats[1]}\n")
            file.write(f"min: {stats[2]}\n")
            file.write(f"max: {stats[3]}\n")
