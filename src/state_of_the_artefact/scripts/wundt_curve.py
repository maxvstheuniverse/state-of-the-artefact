import numpy as np
import matplotlib.pyplot as plt


from state_of_the_artefact.utilities import hedonic, reward, punish

xs = np.linspace(0, 1, 200)

yr = np.array([reward(x) for x in xs])
yp = np.array([punish(x) for x in xs])
ys = np.array([hedonic(x) for x in xs])

plt.plot(xs, yr, color='g', alpha=0.5)
plt.plot(xs, yp, color='r', alpha=0.5)
plt.plot(xs, ys, color='b')

plt.gca().grid(True)
plt.show()
