import numpy as np
import matplotlib.pyplot as plt


from state_of_the_artefact.utilities import hedonic, reward, punish

xs = np.linspace(0, 1, 200)

yr = np.array([reward(x, 1, 5) for x in xs])
yp = np.array([punish(x, 1.05, 10) for x in xs])
ys = np.array([hedonic(x, (1, 1.05), (5, 10)) for x in xs])

plt.plot(xs * 28, yr, color='g', alpha=0.5)
plt.plot(xs * 28, yp, color='r', alpha=0.5)
plt.plot(xs * 28, ys, color='b')

plt.gca().grid(True)
plt.show()
