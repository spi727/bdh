import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loadres import load_all

df = load_all()

# ---- table 1: 3 seeds, 2000 steps, context 256 ----
main = df[(df.max_iters == 2000) & (df.block == 256)]
lines = ["| Model | Val loss (mean ± std) | Bits/byte | ms/step | Train time (s) | Peak MB | Seeds |",
         "|---|---|---|---|---|---|---|"]
for m, g in main.groupby("model"):
    lines.append(f"| {m} | {g.val_loss.mean():.3f} ± {g.val_loss.std():.3f} | {g.val_bpc.mean():.3f} | "
                 f"{g.ms_per_step.mean():.0f} | {g.total_train_s.mean():.0f} | {g.peak_MB.mean():.0f} | {len(g)} |")

# ---- table 2: Transformer with more steps (single seed) ----
long = df[(df.model == "transformer") & (df.max_iters >= 2000) & (df.block == 256) & (df.seed == 1337)]
lines += ["", "Transformer with more steps (seed 1337):", "",
          "| Steps | Val loss | Train time (s) |", "|---|---|---|"]
for _, r in long.sort_values("max_iters").drop_duplicates("max_iters").iterrows():
    lines.append(f"| {int(r.max_iters)} | {r.val_loss:.3f} | {r.total_train_s:.0f} |")
open("report_table.md", "w").write("\n".join(lines))
print("\n".join(lines))

# ---- learning curves (seed 1337) ----
files = {"BDH (2000 steps)": "curve_bdh_s1337.csv",
         "Mamba (2000 steps)": "curve_mamba_s1337.csv",
         "Transformer (2000 steps)": "curve_transformer_s1337_2000.csv",
         "Transformer (8000 steps)": "curve_transformer_s1337_8000.csv"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for label, f in files.items():
    c = pd.read_csv(f, header=None, names=["model", "lr", "step", "sec", "val"])
    c = c[c.step >= 250]      # drops stray step-59 rows left by the 60-step context tests
    ls = "--" if "8000" in label else "-"
    ax[0].plot(c.step, c.val, ls, marker="o", label=label)
    ax[1].plot(c.sec, c.val, ls, marker="o", label=label)
ax[0].set_xlabel("training step"); ax[1].set_xlabel("training time (s)")
for a in ax:
    a.set_ylabel("validation loss"); a.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout(); plt.savefig("final_curves.png", dpi=150)
print("saved final_curves.png")