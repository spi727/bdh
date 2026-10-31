import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

cols = ["model", "lr", "step", "sec", "val"]
df = pd.concat([pd.read_csv(f, header=None, names=cols) for f in glob.glob("curve_*.csv")])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (m, lr), g in df.groupby(["model", "lr"]):
    ax[0].plot(g.step, g.val, marker="o", label=f"{m} lr={lr}")
    ax[1].plot(g.sec, g.val, marker="o", label=f"{m} lr={lr}")
ax[0].set_xlabel("training step"); ax[1].set_xlabel("training time (s, evaluation excluded)")
for a in ax:
    a.set_ylabel("validation loss (nats/byte)"); a.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout(); plt.savefig("curves.png", dpi=150)
print("saved curves.png")