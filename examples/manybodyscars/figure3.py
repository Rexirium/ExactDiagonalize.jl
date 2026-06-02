import numpy as np
import csv
import matplotlib.pyplot as plt

data1 = np.load("examples/manybodyscars/spin1xy_Sz=1_Lmax=18.npz")

spinxy_ts = data1["entropies"][:, -1]
ts = data1["ts"]
Ls = data1["Ls"]

pxp_data = []
with open("examples/manybodyscars/pxp_L18_entropy_dynamics_t0_120_dt0p1.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    
    for i, row in enumerate(reader):
        if i == 0: continue
        pxp_data.append(float(row[1]))

pxp_ts = np.array(pxp_data)

spinxy_Ls = data1["entropies"][-1, :]
pxp_Ls = [0.75855, 0.96097, 1.13899, 1.33447, 1.52293, 1.71432, 1.94539]

k_pxp, b_pxp = np.polyfit(Ls, pxp_Ls, deg=1)
k_sxy, b_sxy = np.polyfit(Ls, spinxy_Ls, deg=1)

Lx = np.linspace(6, 18, 121)
ys_pxp = k_pxp * Lx + b_pxp
ys_sxy = k_sxy * Lx + b_sxy

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "font.size": 14, 
    "xtick.direction": "in",
    "ytick.direction": "in", 
    "legend.frameon": False,
    "legend.edgecolor": 'none'
})

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(ts, spinxy_ts, label="spin-1 XY")
axs[0].plot(ts, pxp_ts, label="static PXP")
axs[0].set(
    xlabel=r"$Jt$", ylabel=r"$S(L/2)$", 
)
axs[0].legend()


axs[1].scatter(Ls, spinxy_Ls, label="spin-1 XY data")
axs[1].plot(Lx, ys_sxy, label="spin-1 XY fit")

axs[1].scatter(Ls, pxp_Ls, label="static PXP data")
axs[1].plot(Lx, ys_pxp, label="static PXP fit")

axs[1].set(
    xlabel=r"$L$", ylabel=r"$S(L/2)$"
)
axs[1].legend(title=r"final time $t = 10^{5}$")

for ax, label in zip(axs, ["a", "b"]):
    ax.text(
        -0.08, 1.07, label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='left'
    )

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
# plt.show()
plt.savefig("examples/manybodyscars/figure3_else.png")


