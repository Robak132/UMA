"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""

from copy import deepcopy
from dataset import Dataset
from model.evolutionary_tree_individual import EvolutionaryTreeIndividual, DivisionNode, NodeType


def test_division_node():
    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])
    node = DivisionNode(xor_dataset.x, xor_dataset.y, 0, 2, None)
    assert node.left.type == NodeType.LEAF_NODE
    assert node.right.type == NodeType.LEAF_NODE


def test_division_node_2():
    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])
    node = DivisionNode(xor_dataset.x, xor_dataset.y, 1, 2, None)
    assert node.left.type == NodeType.DIVISION_NODE
    assert node.right.type == NodeType.DIVISION_NODE


def test_mutation():
    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    tree = EvolutionaryTreeIndividual(xor_dataset.x, xor_dataset.y, 0.5, 1)
    mutated_tree = deepcopy(tree)
    mutated_tree.root.left.mutate_division()

    assert tree.root.left != mutated_tree.root.left
