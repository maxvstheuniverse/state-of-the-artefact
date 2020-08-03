import numpy as np
import random


def generate_midi_data(size, timesteps, midi_range=range(24, 36), probablities=None):
    """ Produces a dataset of midi sequences.

        size is the number of samples.
        timesteps is the length of each sample.
        midi_range is the notes that are available

        x is a reversed sequence, and inputs for the autoencoder.
        y is the original sequence, and the target for the autoencoder.
    """
    x = []
    seen = set()

    characters = [f"{pitch}" for pitch in midi_range]
    _, _, vectorize = create_ctable(characters)

    while len(x) < size:
        tune = np.random.choice(characters, size=timesteps, p=probablities)

        key = ''.join(tune)
        if key in seen:
            continue
        seen.add(key)

        x.append(tune)

    return vectorize(x)


def generate_midi_data_tonic(size, timesteps, key="C", midi_range=range(24, 36), probablities=None):
    x = []
    characters = [f"{pitch}" for pitch in midi_range]
    seen = set()

    # C C# D D# E F F# G G# A A# B

    # diatonic
    # C: 1 0 1 0 1 1 0 1 0 1 0 1
    # F: T T T S T T S

    # pentatonic
    # C: 1 0 1 0 1 0 0 1 0 1 0 0
    # F: 1 0 1 0 0 1 0 1 0 1 0 0

    if key == "C":
        probablities = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7

    if key == "C#":
        probablities = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]) / 7

    if key == "F":
        probablities = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]) / 7

    while len(x) < size:
        tune = np.random.choice(characters, size=timesteps, p=probablities)

        key = ''.join(tune)
        if key in seen:
            continue
        seen.add(key)

        x.append(tune)

    return x


def create_ctable(characters):
    """ Creates two functions, that encode and decode a one-hot encoded sequence
        to and from the given character set.
    """

    def encode(sequence):
        """ One-hot encodes a single sequence. """
        encoded_s = []
        for sc in sequence:
            enc = [1 if sc == c else 0 for i, c in enumerate(characters)]
            encoded_s.append(enc)

        return encoded_s

    def decode(sequence, calc_argmax=True):
        """ Takes a single one-hot encoded sequence. """

        if calc_argmax:
            sequence = sequence.argmax(axis=-1)

        return [characters[i] for i in sequence]

    def vectorize(data):
        """ One-hot encode a dataset, using the provided encode function. """
        return np.array([encode(sequence) for sequence in data], dtype='float32')

    return encode, decode, vectorize
