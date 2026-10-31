import pandas as pd

df = pd.read_csv("results.csv")
df = df[df.Max_Iters == 2000]
g = df.groupby("model").agg(
    runs=("Seed", "count"),
    val_mean=("val_loss", "mean"),
    val_std=("val_loss", "std"),
    bpc_mean=("val_bpc", "mean"),
    ms_step=("ms_per_step", "mean"),
    train_s=("total_train_s", "mean"),
    peak_MB=("peak_MB", "mean"),
).round(3)
print(g.sort_values("val_mean"))
print()
print(df.pivot(index="Seed", columns="model", values="val_loss"))