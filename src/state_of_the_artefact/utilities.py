import numpy as np
import scipy.stats as ss
from scipy.spatial.kdtree import KDTree
from sklearn.neighbors import BallTree


default_eye = np.eye(12)


def reverse_sequences(sequences):
    ndims = len(np.array(sequences).shape)
    assert ndims == 3, f"Expected ndim=3, but got ndim={ndims}"
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


def one_hot(x, num_classes=12):
    if num_classes != 12:
        return np.eye(num_classes)[x]
    return default_eye[x]


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


def hedonic(x, scale=(1, 1.1), offset=(5, 10)):
    return reward(x, scale[0], offset[0]) + punish(x, scale[1], offset[1])


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
    density = frequency / radius ** n

    if return_tree:
        return np.mean(density), tree

    return np.mean(density)


def nn_density(x, tree, r=.1, apply_mean=False):
    frequency = tree.query_radius(x, r, count_only=True)
    density = frequency / r

    if apply_mean:
        return np.mean(density)

    return density


def kde_density(x, tree, apply_mean=False):
    data = tree.get_arrays()[0]
    h = np.std(data) * (4 / 3 / len(data)) ** (1 / 5)  # Silverman's Rule of Thumb
    density = tree.kernel_density(x, h)

    if apply_mean:
        return np.mean(density)

    return density
