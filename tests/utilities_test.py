import numpy as np

from state_of_the_artefact.utilities import reverse_sequences, make_onehot


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


def test_reverse_sequences_one_hot_single():
    a = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0]])
    a = a[::-1]
    b = np.array([[0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]])

    assert np.array_equal(a, b), "The single one-hot sequence are not correctly reversed"


def test_make_onehot():
    a = make_onehot(np.array([[[1, 2, 3, 4], [1, 4, 3, 2]], [[1, 2, 3, 4], [1, 4, 3, 2]]]))
    b = [[[0, 0, 0, 1], [0, 1, 0, 0]], [[0, 0, 0, 1], [0, 1, 0, 0]]]

    assert np.array_equal(a, b), "The 3D array are not correctly onehot encoded"
