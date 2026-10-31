import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df = df[df.Max_Iters == 60].sort_values("block")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m, g in df.groupby("model"):
    ok = g[g.peak_MB <= 4096]
    ax[0].plot(ok.block, ok.ms_per_step, marker="o", label=m)
    ax[1].plot(g.block, g.peak_MB, marker="o", label=m)
ax[0].set_ylabel("ms per step (only runs that fit in 4 GB)")
ax[1].set_ylabel("peak GPU memory (MB)")
ax[1].axhline(4096, ls="--", color="gray")
ax[1].text(130, 4200, "4 GB limit", color="gray")
ax[0].set_yscale("log")
for a in ax:
    a.set_xscale("log", base=2); a.set_xlabel("context length (bytes)")
    a.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout(); plt.savefig("context_scaling.png", dpi=150)
print("saved context_scaling.png")