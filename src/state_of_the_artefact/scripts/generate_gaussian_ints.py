import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

start = 0
x = np.arange(0, 88)
xU, xL = x + .5, x - .5
prob = ss.norm.cdf(xU, loc=24 + start, scale=.3) - ss.norm.cdf(xL, loc=32 + start, scale=.3)
prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
nums = np.random.choice(x, size=100000, p=prob)

xU2, xL2 = x + .5, x - .5
prob2 = ss.norm.cdf(xU2, loc=27 + start, scale=.3) - ss.norm.cdf(xL2, loc=35 + start, scale=.3)
prob2 = prob2 / prob2.sum()  # normalize the probabilities so their sum is 1
nums2 = np.random.choice(x, size=100000, p=prob2)

# plt.hist(nums, density=True, bins=x)
# plt.hist(nums2, density=True, bins=x)

plt.plot(prob, lw=2, label='pdf')
plt.plot(prob2, lw=2, label='pdf2')

plt.gca().grid(True)
plt.gca().set_xticks(x)

plt.gca().set_xlim(20, 40)
plt.gca().set_xlabel("pitches")
plt.gca().set_ylabel("probability")
plt.gca().set_ylim(0, .2)
plt.show()

