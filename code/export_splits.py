# -*- coding: utf-8 -*-
import json, io, os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
os.environ.setdefault("TAEKWONDO_DATA_ROOT", r"E:/yanjiusheng/datas/datas/taekwondo-accelerometer-dataset/data")

base = r"E:\yanjiusheng\datas\datas\taekwondo-accelerometer-dataset"
nb = json.load(open(base + r"\code\taekwondo_recognition.ipynb", encoding="utf-8"))
cells = nb["cells"]
def cellsrc(i):
    src = "".join(cells[i]["source"])
    src = "\n".join([l for l in src.splitlines() if not l.strip().startswith("%")])
    return src.replace("from IPython.display import display", "def display(*args, **kwargs): pass")

g = {}
for i in [1, 3, 9, 11]:
    exec(cellsrc(i), g)

F, y, meta, splits = g["F"], g["y"], g["meta"], g["splits"]

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

rows = []
for sp in splits:
    tr = meta["person_id"].isin(sp["train"]).to_numpy()
    groups_tr = meta.loc[tr, "person_id"].to_numpy()
    Xtr, ytr = F[tr], y[tr]

    # ML inner: StratifiedGroupKFold(5, shuffle=True, random_state=0)
    try:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for k, (itr, iva) in enumerate(skf.split(Xtr, ytr, groups=groups_tr)):
            tr_p = sorted(int(x) for x in np.unique(groups_tr[itr]))
            va_p = sorted(int(x) for x in np.unique(groups_tr[iva]))
            rows.append({"outer_fold": sp["fold"], "method": "ml", "inner_fold": k,
                         "train_participants": ";".join(map(str, tr_p)),
                         "val_participants": ";".join(map(str, va_p))})
    except ValueError:
        # fallback GroupKFold if stratification infeasible
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        for k, (itr, iva) in enumerate(gkf.split(Xtr, ytr, groups=groups_tr)):
            tr_p = sorted(int(x) for x in np.unique(groups_tr[itr]))
            va_p = sorted(int(x) for x in np.unique(groups_tr[iva]))
            rows.append({"outer_fold": sp["fold"], "method": "ml", "inner_fold": k,
                         "train_participants": ";".join(map(str, tr_p)),
                         "val_participants": ";".join(map(str, va_p))})

    # DL inner: GroupShuffleSplit(1, test_size=0.15, random_state=42)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    itr, iva = next(gss.split(Xtr, ytr, groups=groups_tr))
    tr_p = sorted(int(x) for x in np.unique(groups_tr[itr]))
    va_p = sorted(int(x) for x in np.unique(groups_tr[iva]))
    rows.append({"outer_fold": sp["fold"], "method": "dl", "inner_fold": 0,
                 "train_participants": ";".join(map(str, tr_p)),
                 "val_participants": ";".join(map(str, va_p))})

df = pd.DataFrame(rows)
out_path = base + r"\results\tables\splits_inner.csv"
df.to_csv(out_path, index=False)
print("splits_inner.csv written:", len(df), "rows")
print(df.head(8).to_string())
