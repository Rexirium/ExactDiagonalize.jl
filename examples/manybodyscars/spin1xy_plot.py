import numpy as np
import matplotlib.pyplot as plt

L = 10
data = np.load(f"examples/manybodyscars/spin1xy_L={L}.npz")

ts = data["ts"]
entropies = data["entropies"]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "xtick.direction": "in",
    "ytick.direction": "in"
})

fig, ax = plt.subplots()

labels = ["Nematic Neel", "Nematic Ferro", "Sz prod"]
for n, label in enumerate(labels):
    ax.plot(ts, entropies[:, n], label=label)
ax.set_xscale("log")
ax.set_xlabel(r"$Jt$")
ax.set_ylabel(r"$S(L/2)$")
ax.set_title("Growth of Entanglement Entropy in Spin-1 XY Model")
ax.legend()
fig.tight_layout()
plt.show()
#plt.savefig("examples/manybodyscars/spin1xy_evolve.png")