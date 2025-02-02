import matplotlib.pyplot as plt
import numpy as np
aa = np.load("fitness_history.npy")
ab = aa.mean(axis=1)
fig, axs = plt.subplots(2)
axs[0].plot(ab)
for t in range(aa.shape[0]):
    for d in range(aa.shape[1]):
        axs[1].scatter(t, aa[t,d])
plt.show()
