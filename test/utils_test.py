"""
Jakub Robaczewski, Michał Matak
UMA 2022
"""


from utils import calculate_accuracy


def test_accuracy():
    assert calculate_accuracy([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert calculate_accuracy([0, 0, 0, 1], [0, 1, 0, 1]) == 0.75
