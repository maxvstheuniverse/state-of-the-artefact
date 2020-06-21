import numpy as np

from state_of_the_artefact.representation import create_ctable

characters = ['1', '2', '3', '4', '5']
encode, decode, vectorize = create_ctable(characters)


def test_encode_ctable():
    a = encode(['1', '2', '4'])
    b = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0]])
    assert np.array_equal(a, b), "Incorrect one-hot encoding"


def test_decode_ctable():
    encoded = np.array([[1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0]])
    a = decode(encoded)
    b = ['1', '4', '3']
    assert np.array_equal(a, b), "Incorrect one-hot decoding"


def test_vectorize():
    a = vectorize([['1', '2', '3'], ['4', '3', '2']])
    b = np.array([[[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0]],
                  [[0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0]]])

    assert a.shape == b.shape, "The vectorized output did not match the input shape"
    assert np.array_equal(a, b), "The resulting encoding is incorrect"
