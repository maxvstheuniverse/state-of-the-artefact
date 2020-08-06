import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from sklearn.neighbors import BallTree
import time


def density(points):
    total_distance = 0
    count = 0

    head = points[0]
    tail = points[1:]

    while len(tail) > 0:
        for x in tail:
            count += 1
            total_distance += np.linalg.norm(head - x)

        head = tail[0]
        tail = tail[1:]

    return count / total_distance


def delaunay_density(points):
    """ returns the mean distances of the edges. """
    tri = Delaunay(points)
    dists = []

    for indices in tri.simplices:
        combinations = [x for x in np.dstack(np.meshgrid(indices, indices)).reshape(-1, 2) if x[0] != x[1]]
        dists += [np.linalg.norm(points[i] - points[j]) for i, j in combinations]

    return np.mean(dists)


def kdtree_density(points, radius, exclude_root=True):
    tree = cKDTree(points)
    neighbours = tree.query_ball_tree(tree, radius)

    assert len(neighbours) == len(points), "Number of points should be equal to the number of neighbour lists."
    frequency = np.array([len(n) for n in neighbours])

    if exclude_root:
        frequency -= 1

    # / radius ** n ? only if radius > 1, otherwise the numbers get really large really fast.
    # see: https://stackoverflow.com/questions/14070565/calculating-point-density-using-python
    #
    # removing radius altogether return show many points are within range. easier to read?
    # but skewing the density as with a low number of points, might be denser but then is lower?
    return np.mean(frequency / radius ** points.shape[-1])


def balltree_density(x, r=.1):
    tree = BallTree(x)
    frequency = tree.query_radius(x, r, count_only=True)
    return np.mean(frequency / r)


def balltree_kde_density(x):
    tree = BallTree(x)
    h = np.std(x) * (4 / 3 / len(x)) ** (1 / 5)  # Silverman's Rule of Thumb
    density = tree.kernel_density(x, h)
    return np.mean(density)


radius = 20
x1 = [np.random.rand(30 + 8 * i, 32) * 10 - 5 for i in range(1, 251)]
x2 = [np.random.rand(30 + 8 * i, 32) * 5 - 2.5 for i in range(1, 1)]

start_time = time.time()
delaun1 = [balltree_density(x, radius) for x in x1]
delaun2 = [balltree_density(x, radius) for x in x2]
print(f"{time.time() - start_time}s")

start_time = time.time()
kdtree1 = [balltree_kde_density(x) for x in x1]
hs1 = [np.std(x) * (4 / 3 / len(x)) ** (1 / 5) for x in x1]
kdtree2 = [balltree_kde_density(x) for x in x2]
hs2 = [np.std(x) * (4 / 3 / len(x)) ** (1 / 5) for x in x2]
print(f"{time.time() - start_time}s")

fig, axs = plt.subplots(1, 2)

axs[0].plot(delaun1, label="less dense")
axs[0].plot(delaun2, label="dense")
axs[0].set_title(f"BallTree (r={radius})")
axs[0].set_xlabel("Samples (30 + 8 x i)")
axs[0].set_ylabel("Density")

axs[1].plot(kdtree1, label="less dense")
axs[1].plot(kdtree2, label="dense")
axs[1].set_title(f"KDE (h_1={np.mean(hs2):0.3f}, h_2={np.mean(hs1):0.3f})")
axs[1].set_xlabel("Samples (30 + 8 x i)")
axs[1].set_ylabel("Density")

# get the for median large sample

# axs[1][0].scatter(x1[0][:, 0], x1[0][:, 1], label="less dense")
# axs[1][0].scatter(x2[0][:, 0], x2[0][:, 1], label="dense")

# axs[1][1].scatter(x1[9][:, 0], x1[9][:, 1], label="less dense")
# axs[1][1].scatter(x2[9][:, 0], x2[9][:, 1], label="dense")

fig.legend(loc="lower center", labels=["less dense", "dense"], ncol=2)
plt.show()
