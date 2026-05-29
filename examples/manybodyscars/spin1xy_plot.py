import numpy as np
import matplotlib.pyplot as plt

data = np.load("examples/manybodyscars/spin1xy_L=8.npz")

ts = data["ts"]
entropies = data["entropies"]

fig, ax = plt.subplots()

labels = ["NN", "NF", "SZ"]
for n, label in enumerate(labels):
    if n == 0: continue
    ax.plot(ts, entropies[:, n], label=label)
ax.set_xscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Entanglement Entropy (mid-cut)")
ax.set_title("Growth of Entanglement Entropy in Spin-1 XY Model")
ax.legend()
fig.tight_layout()
plt.show()