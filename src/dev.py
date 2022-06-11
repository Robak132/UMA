from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier
from model.evolutionary_tree_individual import EvolutionaryTreeIndividual
from model.decision_tree_classifier import DecisionTreeClassifier
from dataset import BreastTissueDataset, CarEvaluationDataset, TitanicDataset
from utils import split_into_train_test, calculate_accuracy
# tree = EvolutionaryTreeClassifier(alpha=100, beta=-1, division_node_prob=0.3, max_depth=50)
import numpy as np

if __name__ == "__main__":
    np.random.seed(46520)
    dataset = BreastTissueDataset("../data/extracted/breast_tissue.csv")
    # dataset = CarEvaluationDataset("../data/extracted/car_evaluation.csv")
    # dataset = TitanicDataset("../data/extracted/titanic.csv")
    # WineDataset("../data/extracted/wine.csv")
    # RedWineQualityDataset("../data/extracted/winequality_red.csv")
    x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, 0.3)
    # tree = EvolutionaryTreeIndividual(x_train, y_train, division_node_prob=0.3, max_depth=50)
    # print(tree.get_nodes())
    lib_classifier = DecisionTreeClassifier()
    lib_classifier.train(x_train, y_train)
    lib_predictions = lib_classifier.predict(x_test)
    lib_accuracy = calculate_accuracy(y_test, lib_predictions)
    evolutionary_config = {
        "alpha": 100,
        "beta": -0.1,
        "max_depth": 50,
        "max_generations": 1000,
        "population_size": 50,
        "crossover": True,
        "division_node_prob": 0.5,
        "mutation_change_type_prob": 0.2,
        "tournament_size": 2,
        "elite_size": 1
    }

    classifier = EvolutionaryTreeClassifier(evolutionary_config)
    classifier.train(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(len(classifier.best_tree.get_nodes()))
    print(f"Accuracy to beat: {lib_accuracy}")
    print(lib_classifier.clf.tree_.node_count)
# self.train(x_train, y_train)
# predictions = self.predict(x_test)
# current_accuracy = calculate_accuracy(y_test, predictions)