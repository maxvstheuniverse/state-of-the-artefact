import numpy as np

from state_of_the_artefact.utilities import reverse_sequences, one_hot


def test_reverse_sequences():
    a = reverse_sequences([[['a', 'b', 'c'], ['c', 'b', 'a']]])
    b = [[['c', 'b', 'a'], ['a', 'b', 'c']]]

    assert np.array_equal(a, b), "The sequences are not correctly reversed"


def test_reverse_sequences_one_hot():
    a = reverse_sequences([[[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0]],
                           [[0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 0]]])
    b = np.array([[[0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0]],
                  [[0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0]]])

    assert np.array_equal(a, b), "The one-hot sequences are not correctly reversed"


def test_one_hot_default():
    array = np.array([[4, 6, 10],
                      [3, 0, 2]])

    a = one_hot(array)
    b = np.array([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                  [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

    assert np.array_equal(a, b), "with default depth, not correctly onehot encoded"


def test_one_hot_5d():
    array = np.array([[4, 4, 1],
                      [3, 0, 2]])

    a = one_hot(array, 5)
    b = np.array([[[0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0]],
                  [[0, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]]])

    assert np.array_equal(a, b), "with depth 5, not correctly onehot encoded"
