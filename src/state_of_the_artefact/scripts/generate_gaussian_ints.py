import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 88)
xU, xL = x + .5, x - .5
prob = ss.norm.cdf(xU, loc=24, scale=3) - ss.norm.cdf(xL, loc=35, scale=3)
prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
nums = np.random.choice(x, size=100000, p=prob)

xU2, xL2 = x + .5, x - .5
prob2 = ss.norm.cdf(xU2, loc=60, scale=3) - ss.norm.cdf(xL2, loc=71, scale=3)
prob2 = prob2 / prob2.sum()  # normalize the probabilities so their sum is 1
nums2 = np.random.choice(x, size=100000, p=prob2)

plt.hist(nums, density=True, bins=x)
plt.hist(nums2, density=True, bins=x)

plt.plot(prob, lw=2, label='pdf')
plt.plot(prob2, lw=2, label='pdf2')

plt.gca().grid(True)
plt.gca().set_xticks(x[::5])
plt.show()

