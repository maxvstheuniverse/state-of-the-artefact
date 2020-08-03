import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, KDTree


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
    tree = KDTree(points)
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


x1 = [np.random.rand(1000, 2) * 10 - 5 for _ in range(50)]
x2 = [np.random.rand(5, 2) * 10 - 5 for _ in range(50)]

delaun1 = [delaunay_density(x) for x in x1]
delaun2 = [delaunay_density(x) for x in x2]

kdtree1 = [kdtree_density(x, 1) for x in x1]
kdtree2 = [kdtree_density(x, 1) for x in x2]

fig, axs = plt.subplots(2, 2)

axs[0][0].plot(delaun1, label="large")
axs[0][0].plot(delaun2, label="small")
axs[0][0].set_title(f"Delaunay (l={np.mean(delaun1):0.3f}, s={np.mean(delaun2):0.3f})\nLower is denser")

axs[0][1].plot(kdtree1, label="large")
axs[0][1].plot(kdtree2, label="small")
axs[0][1].set_title(f"KDTree (l={np.mean(kdtree1):0.3f}, s={np.mean(kdtree2):0.3f})\nHigher is denser")

# get the for median large samples
i1 = np.argsort(delaun1)[len(delaun1) // 2]
i2 = np.argsort(kdtree1)[len(kdtree1) // 2]

axs[1][0].scatter(x1[i1][:, 0], x1[i1][:, 1])
axs[1][0].scatter(x2[i1][:, 0], x2[i1][:, 1])
axs[1][0].set_title("Delaunay Median Sample")

axs[1][1].scatter(x1[i2][:, 0], x1[i2][:, 1])
axs[1][1].scatter(x2[i2][:, 0], x2[i2][:, 1])
axs[1][1].set_title("KDTree Median Sample")

plt.show()
