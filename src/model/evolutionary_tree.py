# from .evolutionary.treeindividual import TreeIndividual
# from utils import calculate_accuracy
#
# class EvolutionaryTreeClassifier:
#     def __init__(self, X, Y):
#         self.tree = TreeIndividual(X, Y)
#
#     def train(self, X, Y):
#         predictions = self.tree.predict(X)
#         print(calculate_accuracy(Y, predictions))
#
#     def predict(self, X):
#         predictions = []
#         for index, row in X.iterrows():
#             prediction = self.tree.predict(row)
#             predictions.append(self.tree.predict(row))
#         return predictions
