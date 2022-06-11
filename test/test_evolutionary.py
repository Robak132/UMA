from dataset import Dataset
from model.evolutionary_tree_individual import EvolutionaryTreeIndividual, DivisionNode, LeafNode


def test_division_node():
    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])
    node = DivisionNode(xor_dataset.x, xor_dataset.y, 0, 20, None)
    assert node.left.__class__ is LeafNode
    assert node.right.__class__ is LeafNode


def test_mutation():
    xor_dataset = Dataset([[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 0, 1])

    tree = EvolutionaryTreeIndividual(xor_dataset.x, xor_dataset.y, 0.5, 0.3)
    pass
