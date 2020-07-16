import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 88)
xU, xL = x + .5, x - .5
prob = ss.norm.cdf(xU, loc=60, scale=6) - ss.norm.cdf(xL, loc=60, scale=6)
prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
nums = np.random.choice(x, size=100000, p=prob)

xU2, xL2 = x + .5, x - .5
prob2 = ss.norm.cdf(xU2, loc=21, scale=6) - ss.norm.cdf(xL2, loc=21, scale=6)
prob2 = prob2 / prob2.sum()  # normalize the probabilities so their sum is 1
nums2 = np.random.choice(x, size=100000, p=prob2)

plt.hist(nums, bins=x)
plt.hist(nums2, bins=x)
plt.show()
# print(nums + 44)

