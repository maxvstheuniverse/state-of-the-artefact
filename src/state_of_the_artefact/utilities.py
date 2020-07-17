import numpy as np
import scipy.stats as ss


def reverse_sequences(sequences):
    return np.array([seq[::-1] for seq in sequences])


def calculate_distances(x):
    cm = [np.linalg.norm(a - positions, axis=1) for a in positions]
    return np.array(cm)


def interpolate(a, b, nsteps):
    """ Interpolates two arrays in the given number of steps. """
    alpha = 1 / nsteps
    steps = []

    for i in range(1, nsteps + 1):
        alpha_star = alpha * i
        step = alpha_star * b + (1 - alpha_star) * a
        steps.append(step)

    return np.array(steps)


def make_onehot(x):
    m = np.zeros(x.shape)
    indices = np.argmax(x, axis=-1)
    for i, n in enumerate(m):
        for j, o in enumerate(n):
            o[indices[i, j]] = 1
    return m


def generate_integer_probabilities(minv, maxv, loc, scale):
    """ Returns a probabilitie distrubtion for the given range of integers. """
    x = np.arange(minv, maxv)
    xUb, xLb = x + 0.5, x - 0.5  # deviate the lower and upperbound for each integer

    if not isinstance(loc, tuple):
        loc = (loc, loc)

    prob = ss.norm.cdf(xUb, loc=loc, scale=scale) - ss.norm.cdf(xLb, loc=loc, scale=scale)
    return prob / prob.sum()  # return normalized probablities
