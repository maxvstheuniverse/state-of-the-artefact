import numpy as np

from state_of_the_artefact.utilities import reverse_sequences


def test_reverse_sequences():
    a = reverse_sequences([['a', 'b', 'c'], ['c', 'b', 'a']])
    b = [['c', 'b', 'a'], ['a', 'b', 'c']]

    assert np.array_equal(a, b), "The sequences are not correctly reversed"