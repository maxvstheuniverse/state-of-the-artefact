import numpy as np
import scipy.stats as ss
from scipy.spatial.kdtree import KDTree


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


def reward(x, scale=1, offset=5):
    return scale * (1 / (1 + np.exp(-15 * x + offset)))


def punish(x, scale=1.1, offset=10):
    return -scale * (1 / (1 + np.exp(-15 * x + offset)))


def hedonic(x):
    return reward(x) + punish(x)


def delaunay_density(points, mode="edges", return_triangles=False):
    """ Returns the density calculated using Delaunay Triangulation.
        Lower is denser.

        If `mode` is `edges`, uses the edges to calculate density. The other option
        is to use the area of triangle. This is not yet implemented.

        If `return_triangles` is `True` returns a tuple of the density and the triangles
    """
    assert mode == "edges", "Other modes not yet available"

    tri = Delaunay(points)
    dists = []

    for indices in tri.simplices:
        if mode == "edges":
            combinations = np.dstack(np.meshgrid(indices, indices)).reshape(-1, 2)
            combinations = [x for x in combinations if x[0] != x[1]]
            dists += [np.linalg.norm(points[i] - points[j]) for i, j in combinations]

        if mode == "area":
            pass

    density = np.mean(dists)

    if return_triangles:
        return density, tri

    return density


def kdtree_density(points, radius, return_tree=False):
    """ Returns the density calculated sing a Ball Tree.
        Higher is denser.

        Scales according to the dimension of the points.
        see: https://stackoverflow.com/questions/14070565/calculating-point-density-using-python

        If `return_tree` is `True` returns a tuple of the density and the KDTree.
    """
    tree = KDTree(points)
    n = points.shape[-1]

    ball_trees = tree.query_ball_tree(tree, radius)
    frequency = np.array([len(neighbours) for neighbours in ball_trees])
    density = np.mean(frequency / radius ** n)

    if return_tree:
        return density, tree

    return density
