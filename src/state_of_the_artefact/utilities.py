import numpy as np


def reverse_sequences(sequences):
    return np.array([seq[::-1] for seq in sequences])


def calculate_distances(x):
    cm = [np.linalg.norm(a - positions, axis=1) for a in positions]
    return np.array(cm)
