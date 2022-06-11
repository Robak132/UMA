from model.evolutionary_tree_classifier import EvolutionaryTreeClassifier
from model.evolutionary_tree_individual import EvolutionaryTreeIndividual
from model.decision_tree_classifier import DecisionTreeClassifier
from dataset import BreastTissueDataset, CarEvaluationDataset, TitanicDataset
from utils import split_into_train_test, calculate_accuracy
# tree = EvolutionaryTreeClassifier(alpha=100, beta=-1, division_node_prob=0.3, max_depth=50)

if __name__ == "__main__":
    dataset = BreastTissueDataset("../data/extracted/breast_tissue.csv")
    # dataset = CarEvaluationDataset("../data/extracted/car_evaluation.csv")
    # dataset = TitanicDataset("../data/extracted/titanic.csv")
    # WineDataset("../data/extracted/wine.csv")
    # RedWineQualityDataset("../data/extracted/winequality_red.csv")
    x_train, x_test, y_train, y_test = split_into_train_test(dataset.x, dataset.y, 0.3)
    # tree = EvolutionaryTreeIndividual(x_train, y_train, division_node_prob=0.3, max_depth=50)
    # print(tree.get_nodes())
    classifier = EvolutionaryTreeClassifier(alpha=100, beta=-0.1, max_generations=500, division_node_prob=0.5, max_depth=50)
    classifier.train(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(len(classifier.best_tree.get_nodes()))
    lib_classifier = DecisionTreeClassifier()
    lib_classifier.train(x_train, y_train)
    lib_predictions = lib_classifier.predict(x_test)
    lib_accuracy = calculate_accuracy(y_test, lib_predictions)
    print(f"Accuracy to beat: {lib_accuracy}")
    print(lib_classifier.clf.tree_.node_count)
# self.train(x_train, y_train)
# predictions = self.predict(x_test)
# current_accuracy = calculate_accuracy(y_test, predictions)