import numpy as np
import matplotlib.pyplot as plt

z = np.random.rand(10, 2)

mu = z.mean(axis=0, keepdims=True)
median = np.median(z, axis=0, keepdims=True)
sigma = z.var(axis=0, keepdims=True, ddof=0)
std = z.std(axis=0, keepdims=True)

print(sigma, std)

x1 = np.random.normal(mu, sigma, (1000, 2))
x2 = np.random.normal(mu, std, (1000, 2))

x1_mu = np.mean(x1, axis=0)
x2_mu = np.mean(x2, axis=0)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(x1[:, 0], x1[:, 1], marker='.', alpha=.3, label="samples")
ax[0].scatter(z[:, 0], z[:, 1], color='red', label="z")
ax[0].scatter(mu[0][0], mu[0][1], label="mean")
ax[0].scatter(median[0][0], median[0][1], c='gold', marker=".", label="median")

ax[1].scatter(x2[:, 0], x2[:, 1], marker='.', alpha=.3, label="samples")
ax[1].scatter(z[:, 0], z[:, 1], color='red', label="z")
ax[1].scatter(mu[0][0], mu[0][1], label="mean")
ax[0].scatter(median[0][0], median[0][1], c='gold', marker='.', label="median")

ax[0].set_title('var')
ax[1].set_title('std')

ax[0].set_ylim(-1, 2)
ax[0].set_xlim(-1, 2)

ax[1].set_ylim(-1, 2)
ax[1].set_xlim(-1, 2)
fig.legend(labels=["samples", "z", "mean", "median"])
plt.show()
