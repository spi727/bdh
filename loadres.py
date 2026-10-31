import csv, os
import pandas as pd

COLS = ["model", "lr", "max_iters", "seed", "block", "M_params", "val_loss",
        "val_bpc", "ms_per_step", "tokens_per_sec", "total_train_s", "peak_MB"]

def load_all():
    rows = []
    for path in ["results_main.csv", "results.csv"]:
        if not os.path.exists(path):
            continue
        for r in csv.reader(open(path)):
            if not r or r[0] == "model":
                continue
            if len(r) == 11:          # old format without the block column
                r.insert(4, "256")
            if len(r) == 12:
                rows.append(r)
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c])
    return df.drop_duplicates().reset_index(drop=True)