import numpy as np
import matplotlib.pyplot as plt

nmax, Lmax = 2, 12
data = np.load(f"examples/manybodyscars/spin1xy_nmax={nmax}_Lmax={Lmax}_scar.npz")

h = data["params"][1]
Ls = data["Ls"]
ts = data["ts"]
entropies = data["entropies"]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "xtick.direction": "in",
    "ytick.direction": "in"
})

fig, ax = plt.subplots(figsize=(8, 4))

for n, L in enumerate(Ls):
    ax.plot(ts, entropies[:, n], label=rf"$L = {L}$")

ax.set_xscale("log")
ax.set_ylim(0.0, np.log(2))
ax.set(xlabel=r"$Jt$", ylabel=r"$S(L/2)$", title="Spin-1 XY chain entanglement entropy")
ax.legend(title=rf"$h = {h}J$")
plt.show()
#plt.savefig(f"examples/manybodyscars/spin1xy_nmax={nmax}_Lmax={Lmax}_scar.png")