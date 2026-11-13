#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable analysis pipeline for the five-body-site Taekwondo dataset.

Auto-generated from code/taekwondo_recognition.ipynb (run top-to-bottom).
Set TAEKWONDO_DATA_ROOT to the data/ directory; outputs go under results/.
Set TAEKWONDO_N_JOBS to control GridSearchCV parallelism (default -1).
"""
import os
import matplotlib
matplotlib.use("Agg")


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
def display(*args, **kwargs): pass


# 数据根目录（可用环境变量 TAEKWONDO_DATA_ROOT 覆盖）
DATA_ROOT = Path(os.environ.get("TAEKWONDO_DATA_ROOT", r"E:\yanjiusheng\datas\腿法\dataset"))
RESULTS_DIR = Path.cwd() / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_TABLES_DIR.mkdir(exist_ok=True)
RESULTS_FIGURES_DIR.mkdir(exist_ok=True)

CHANNELS = [
    "x_W", "y_W", "z_W", "x_rL", "y_rL", "z_rL", "x_rH", "y_rH", "z_rH",
    "x_lL", "y_lL", "z_lL", "x_lH", "y_lH", "z_lH",
]
SENSOR_GROUPS = {
    "waist": ["x_W", "y_W", "z_W"],
    "ankles": ["x_rL", "y_rL", "z_rL", "x_lL", "y_lL", "z_lL"],
    "wrists": ["x_rH", "y_rH", "z_rH", "x_lH", "y_lH", "z_lH"],
}
KICK2EN = {
    "左前踢": "Left Front Kick", "右前踢": "Right Front Kick",
    "左横踢": "Left Roundhouse Kick", "右横踢": "Right Roundhouse Kick",
    "左侧踢": "Left Side Kick", "右侧踢": "Right Side Kick",
    "左后踢": "Left Back Kick", "右后踢": "Right Back Kick",
}
CLASSES = list(KICK2EN.values()) + [
    "Left Straight Punch", "Right Straight Punch",
    "Left High Block", "Right High Block",
    "Left Middle Block", "Right Middle Block",
    "Left Low Block", "Right Low Block",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

FS = 30.0  # 采样率 Hz

# 【可调参数】
RANDOM_STATE = 42
SVM_PARAM_GRID = {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
RF_PARAM_GRID = {"n_estimators": [50, 100, 200], "max_depth": [None, 8, 16, 32], "min_samples_split": [2, 5]}
DT_PARAM_GRID = {"max_depth": [None, 4, 8, 16], "min_samples_split": [2, 5]}
GRID_CV = 5
RF_N_ESTIMATORS = 100
DL_EPOCHS = 30
DL_BATCH_SIZE = 32
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_HIDDEN = 64
DL_DROPOUT = 0.3
DL_LR_STEP = 4
DL_LR_GAMMA = 0.9

def read_table(path):
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        header = [str(h).strip() for h in rows[0]]
        return pd.DataFrame(rows[1:], columns=header)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def parse_id(id_str):
    action_raw, subject_id, trial = id_str.rsplit("_", 2)
    return action_raw, subject_id, trial


def load_dataset(root=DATA_ROOT):
    def _table(region, kind):
        if region == "kick":
            folder = "kick technique"
            data_n = ["data_kick_technique.xlsx", "data_kick_technique.csv"]
            label_n = ["label_kick technique.xlsx", "label_kick_technique.xlsx",
                       "label_kick technique.csv", "label_kick_technique.csv"]
        else:
            folder = "punch technique"
            data_n = ["data_punch technique.xlsx", "data_punch_technique.xlsx",
                      "data_punch technique.csv", "data_punch_technique.csv"]
            label_n = ["label_punch_technique.xlsx", "label_punch_technique.csv"]
        for name in (data_n if kind == "data" else label_n):
            for cand in (root / folder / name, root / name):
                if cand.exists():
                    return cand
        raise FileNotFoundError(f"cannot find {region} {kind} table under {root}")
    files = {r: (_table(r, "data"), _table(r, "label")) for r in ("kick", "punch")}
    parts = []
    for region, (data_path, label_path) in files.items():
        data = read_table(data_path)
        label = read_table(label_path)
        data["id"] = data["id"].astype(str).str.strip()
        label["id"] = label["id"].astype(str).str.strip()
        assert set(data["id"].unique()) == set(label["id"].unique()), f"{region} id 不一致"
        parsed = data["id"].map(parse_id)
        action_raw = parsed.map(lambda t: t[0])
        subject = parsed.map(lambda t: t[1])
        trial = parsed.map(lambda t: int(t[2]))
        action = action_raw.map(lambda a: KICK2EN.get(a, a)) if region == "kick" else action_raw
        sid = subject.astype(int)
        person_id = sid  # unified participant id 1..20 (punch data remapped to 1..15)
        df = data.copy()
        df["subject_key"] = region + "_" + subject
        df["person_id"] = person_id
        df["subject_id"] = subject
        df["region"] = region
        df["action"] = action
        df["trial"] = trial
        df["time"] = pd.to_numeric(df["time"]).astype(int)
        for c in CHANNELS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        parts.append(df[["id", "subject_key", "person_id", "subject_id", "region", "action", "trial", "time"] + CHANNELS])
    merged = pd.concat(parts, ignore_index=True)
    assert merged["action"].isin(CLASSES).all(), "存在未映射到 16 类的动作"
    return merged


def to_windows(df, channels=CHANNELS):
    df = df.sort_values(["id", "time"], kind="stable")
    n = len(df["id"].unique())
    arr = df[channels].to_numpy(dtype=np.float32)
    X = arr.reshape(n, 120, len(channels)).transpose(0, 2, 1)
    first = df.drop_duplicates("id")
    y = np.array([CLASS2IDX[a] for a in first["action"]], dtype=np.int64)
    meta = first[["id", "subject_key", "person_id", "region", "action", "trial"]].reset_index(drop=True)
    return X, y, meta


df = load_dataset()
X, y, meta = to_windows(df)
print(f"样本 {len(y)}，输入形状 {X.shape}，类别 {len(CLASSES)}")
display(df.drop_duplicates("id").groupby("action").size().to_frame("n_samples"))

G_LIMIT = 8.0  # ±8g 饱和阈值


def channel_statistics(df):
    rows = []
    for c in CHANNELS:
        x = df[c].to_numpy(dtype=np.float64)
        mean, std = x.mean(), x.std(ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (x - mean) / std if std > 0 else np.zeros_like(x)
            skew = float(np.mean(z ** 3)) if std > 0 else 0.0
            kurt = float(np.mean(z ** 4) - 3.0) if std > 0 else 0.0
        rows.append({
            "channel": c, "min": float(x.min()), "max": float(x.max()),
            "mean": float(mean), "std": float(std), "skew": skew, "kurtosis": kurt,
            "peak": float(np.max(np.abs(x))),
            "n_abs_ge_8g": int((np.abs(x) >= G_LIMIT).sum()),
            "frac_abs_ge_8g": float((np.abs(x) >= G_LIMIT).mean()),
        })
    return pd.DataFrame(rows)


def class_counts(df):
    return df.drop_duplicates("id").groupby("action").size().rename("n_samples").reset_index()


stats = channel_statistics(df)
counts = class_counts(df)
stats.to_csv(RESULTS_TABLES_DIR / "channel_statistics.csv", index=False)
counts.to_csv(RESULTS_TABLES_DIR / "class_counts.csv", index=False)
display(stats.round(4))
print(f"饱和(≥8g)最大比例 = {stats['frac_abs_ge_8g'].max():.4f}")

GROUPS = {
    "Front Kick": ["Left Front Kick", "Right Front Kick"],
    "Roundhouse Kick": ["Left Roundhouse Kick", "Right Roundhouse Kick"],
    "Side Kick": ["Left Side Kick", "Right Side Kick"],
    "Back Kick": ["Left Back Kick", "Right Back Kick"],
}


def resultant_magnitude(x):
    return np.sqrt(np.sum(x ** 2, axis=0))


def power_spectrum(sig, fs=FS):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    p = np.abs(np.fft.rfft(sig)) ** 2 / n
    return f, p


def energy_below(f, p, cutoff=15.0):
    return float(p[f <= cutoff].sum() / (p.sum() + 1e-12))


def group_spectra(meta, X):
    result = {}
    for gname, actions in GROUPS.items():
        idx = [i for i, a in enumerate(meta["action"]) if a in actions]
        f, _ = power_spectrum(np.zeros(120))
        ps = np.zeros_like(f)
        fracs = []
        for i in idx:
            _, p = power_spectrum(resultant_magnitude(X[i]))
            ps += p / (p.sum() + 1e-12)
            fracs.append(energy_below(f, p))
        result[gname] = (f, ps / len(idx), float(np.mean(fracs)), len(idx))
    return result


spec = group_spectra(meta, X)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (gname, (f, ps, frac, n)) in zip(axes.ravel(), spec.items()):
    ax.plot(f, ps)
    ax.axvline(15.0, color="r", ls="--", label="15 Hz")
    ax.set_title(f"{gname} (n={n}, <15Hz={frac:.3f})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.legend()
fig.suptitle("Mean power spectrum per kick group")
fig.tight_layout()
fig.savefig(RESULTS_FIGURES_DIR / "spectral_analysis.png", dpi=150)
plt.show()

def get_person_ids(df):
    return sorted(df["person_id"].unique().tolist())


def make_splits(person_ids, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    both = sorted(p for p in person_ids if p <= 15)        # 双腿法拳法（15 人）
    kick_only = sorted(p for p in person_ids if p >= 16)   # 只做腿法（5 人）
    both = [int(x) for x in rng.permutation(both)]
    kick_only = [int(x) for x in rng.permutation(kick_only)]
    splits = []
    fold = 0
    for i, ko in enumerate(kick_only):
        test = [ko, both[i]]
        train = [p for p in person_ids if p not in set(test)]
        splits.append({"fold": fold, "train": train, "test": test})
        fold += 1
    rest = both[len(kick_only):]
    for i in range(0, len(rest), 2):
        test = rest[i:i + 2]
        train = [p for p in person_ids if p not in set(test)]
        splits.append({"fold": fold, "train": train, "test": test})
        fold += 1
    return splits


persons = get_person_ids(df)
splits = make_splits(persons)
print(f"真实被试 {len(persons)} 人：{persons}")
for sp in splits:
    print(f"折 {sp['fold']}: test={sp['test']}  train_n={len(sp['train'])}")

# 导出固定外层划分（可复现）
_splits_out = pd.DataFrame([
    {"fold": sp["fold"], "test": ";".join(map(str, sp["test"])), "train": ";".join(map(str, sp["train"]))}
    for sp in splits
])
_splits_out.to_csv(RESULTS_TABLES_DIR / "splits_outer.csv", index=False)
print("外层固定划分已导出 -> results/tables/splits_outer.csv")

FEATURE_NAMES = ["mean", "std", "var", "min", "max", "skew", "kurtosis",
                 "rms", "zcr", "fft_domfreq", "fft_energy", "fft_entropy"]


def extract_features(X):
    n, C, T = X.shape
    feat = np.zeros((n, C, len(FEATURE_NAMES)), dtype=np.float64)
    feat[:, :, 0] = X.mean(axis=2)
    feat[:, :, 1] = X.std(axis=2)
    feat[:, :, 2] = X.var(axis=2)
    feat[:, :, 3] = X.min(axis=2)
    feat[:, :, 4] = X.max(axis=2)
    feat[:, :, 7] = np.sqrt(np.mean(X ** 2, axis=2))
    feat[:, :, 8] = np.sum((X[:, :, :-1] >= 0) != (X[:, :, 1:] >= 0), axis=2) / T
    mean = feat[:, :, 0:1]
    std = feat[:, :, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (X - mean) / np.where(std > 0, std, 1.0), 0.0)
    feat[:, :, 5] = np.mean(z ** 3, axis=2)
    feat[:, :, 6] = np.mean(z ** 4, axis=2) - 3.0
    spec = np.abs(np.fft.rfft(X, axis=2)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    feat[:, :, 9] = freqs[np.argmax(spec, axis=2)]
    feat[:, :, 10] = spec.sum(axis=2)
    ps = spec / (spec.sum(axis=2, keepdims=True) + 1e-12)
    feat[:, :, 11] = -np.sum(ps * np.log2(ps + 1e-12), axis=2)
    return feat.reshape(n, C * len(FEATURE_NAMES))


F = extract_features(X)
print(f"特征矩阵 {F.shape}（样本 × 特征）")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# ===================== 全局变量（请自行修改为你的真实配置） =====================


def build_model(model_type, class_weight=None):
    kw = dict(random_state=0)
    if class_weight:
        kw["class_weight"] = class_weight
    if model_type == "svm":
        return SVC(**kw)
    if model_type == "rf":
        return RandomForestClassifier(**kw)
    if model_type == "dt":
        return DecisionTreeClassifier(**kw)
    raise ValueError(model_type)

def run_cv(model_type, param_grid, F, y, meta, splits, class_weight=None, gs_scoring="f1_macro"):
    fold_rows = []          # 每折整体指标
    fold_perclass_list = [] # 每折每个类别的precision/recall/f1
    fold_best_params = []   # 每折网格搜索最优参数
    yt_all, yp_all = [], []
    for sp in splits:
        tr = meta["person_id"].isin(sp["train"]).to_numpy()
        te = meta["person_id"].isin(sp["test"]).to_numpy()
        Xtr, Xte = F[tr], F[te]
        ytr, yte = y[tr], y[te]
        groups_tr = meta.loc[tr, "person_id"].to_numpy()  # 每个训练样本对应的被试 id

        # 标准化放入 pipeline：仅在内层训练子集上拟合，避免内层验证样本泄漏
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", build_model(model_type, class_weight=class_weight)),
        ])
        param_grid_piped = {f"model__{k}": v for k, v in param_grid.items()}
        # 内层按被试分组的交叉验证（被试不相交）
        inner_cv = StratifiedGroupKFold(n_splits=GRID_CV, shuffle=True, random_state=0)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid_piped,
            cv=inner_cv,
            scoring=gs_scoring,
            n_jobs=int(os.environ.get("TAEKWONDO_N_JOBS", "-1")),
            refit=True
        )
        try:
            gs.fit(Xtr, ytr, groups=groups_tr)
        except ValueError:
            inner_cv = GroupKFold(n_splits=GRID_CV)
            gs = GridSearchCV(pipe, param_grid_piped, cv=inner_cv,
                              scoring=gs_scoring, n_jobs=int(os.environ.get("TAEKWONDO_N_JOBS", "-1")), refit=True)
            gs.fit(Xtr, ytr, groups=groups_tr)
        pred = gs.predict(Xte)

        # 本折整体指标
        acc = accuracy_score(yte, pred)
        macro_f1 = f1_score(yte, pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(yte, pred, average="weighted", zero_division=0)
        w_prec = precision_score(yte, pred, average="weighted", zero_division=0)
        w_rec = recall_score(yte, pred, average="weighted", zero_division=0)

        fold_rows.append({
            "fold": sp["fold"],
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "weighted_precision": w_prec,
            "weighted_recall": w_rec,
        })
        fold_best_params.append({"fold": sp["fold"], "best_params": gs.best_params_})

        # 本折每一类 precision / recall / f1
        rep_fold = classification_report(
            yte, pred,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        )
        for cls_name in CLASSES:
            fold_perclass_list.append({
                "fold": sp["fold"],
                "class": cls_name,
                "precision": rep_fold[cls_name]["precision"],
                "recall": rep_fold[cls_name]["recall"],
                "f1": rep_fold[cls_name]["f1-score"]
            })
        yt_all.append(yte)
        yp_all.append(pred)

    # 全部折合并后的全局指标（用于画混淆矩阵，不做为主结果）
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    agg = {
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "weighted_precision": precision_score(yt, yp, average="weighted", zero_division=0),
        "weighted_recall": recall_score(yt, yp, average="weighted", zero_division=0),
    }
    cm = confusion_matrix(yt, yp, labels=range(len(CLASSES)))
    rep_global = classification_report(
        yt, yp, labels=range(len(CLASSES)),
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    df_global_perclass = pd.DataFrame(rep_global).T
    df_folds = pd.DataFrame(fold_rows)
    df_fold_perclass = pd.DataFrame(fold_perclass_list)
    df_bestparams = pd.DataFrame(fold_best_params)
    return agg, cm, df_folds, df_global_perclass, df_fold_perclass, df_bestparams

# ===================== 主循环 =====================
ml_results = {}
model_configs = [
    ("svm", SVM_PARAM_GRID),
    ("rf", RF_PARAM_GRID),
    ("dt", DT_PARAM_GRID)
]
for m_name, grid in model_configs:
    agg, cm, folds_df, pc_global_df, pc_fold_df, bestparam_df = run_cv(
        model_type=m_name,
        param_grid=grid,
        F=F,
        y=y,
        meta=meta,
        splits=splits,
        class_weight=None,
        gs_scoring="f1_macro"
    )
    ml_results[m_name] = agg
    np.save(RESULTS_DIR / f"ml_{m_name}_confusion.npy", cm)
    folds_df.to_csv(RESULTS_TABLES_DIR / f"ml_{m_name}_folds.csv", index=False)
    pc_global_df.to_csv(RESULTS_TABLES_DIR / f"ml_{m_name}_per_class_global.csv", index=True)
    pc_fold_df.to_csv(RESULTS_TABLES_DIR / f"ml_{m_name}_per_class_folds.csv", index=False)
    bestparam_df.to_csv(RESULTS_TABLES_DIR / f"ml_{m_name}_best_params.csv", index=False)

# ========== 全部模型汇总：k折指标 mean ± std（论文主表，推荐） ==========
summary_mean = []
for m_name in ["svm","rf","dt"]:
    df_fold = pd.read_csv(RESULTS_TABLES_DIR / f"ml_{m_name}_folds.csv")
    stat = df_fold[["accuracy","weighted_precision","weighted_recall","weighted_f1","macro_f1"]].agg(["mean","std"])
    summary_mean.append({
        "model": m_name,
        "accuracy_mean": stat.loc["mean","accuracy"],
        "accuracy_std": stat.loc["std","accuracy"],
        "wp_mean": stat.loc["mean","weighted_precision"],
        "wp_std": stat.loc["std","weighted_precision"],
        "wr_mean": stat.loc["mean","weighted_recall"],
        "wr_std": stat.loc["std","weighted_recall"],
        "wf1_mean": stat.loc["mean","weighted_f1"],
        "wf1_std": stat.loc["std","weighted_f1"],
        "mf1_mean": stat.loc["mean","macro_f1"],
        "mf1_std": stat.loc["std","macro_f1"],
    })
summary_df_mean = pd.DataFrame(summary_mean)
summary_df_mean.to_csv(RESULTS_TABLES_DIR / "ml_all_model_summary_mean_std.csv", index=False)
print("\n==== 所有模型汇总（k折交叉验证 mean ± std） ====")
print(summary_df_mean.round(4))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import random

# ===================== 【全局超参数】 =====================
# CLASSES 已在 Setup 定义（16 类英文名）
DL_HIDDEN = 64
DL_EPOCHS = 30
DL_BATCH_SIZE = 32
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_LR_STEP = 4
DL_LR_GAMMA = 0.9
DL_DROPOUT = 0.3
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_TABLES_DIR.mkdir(exist_ok=True)
RESULTS_FIGURES_DIR.mkdir(exist_ok=True)

# ===================== 固定随机种子（绝对可复现核心） =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # 可选，报错就注释掉

# ===================== 模型定义 =====================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride), nn.BatchNorm1d(out_ch))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self, in_channels=15, n_classes=16, hidden=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, 2, 3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1),
        )
        self.block1 = ResidualBlock(hidden, hidden, dropout=dropout)
        self.block2 = ResidualBlock(hidden, hidden * 2, stride=2, dropout=dropout)
        self.block3 = ResidualBlock(hidden * 2, hidden * 2, dropout=dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

class GRUModel(nn.Module):
    def __init__(self, in_channels=15, n_classes=16, hidden=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=True)
        self.head = nn.Linear(hidden * 2, n_classes)
    def forward(self, x):
        out, _ = self.gru(x.transpose(1, 2))
        return self.head(out[:, -1])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerModel(nn.Module):
    def __init__(self, in_channels=15, n_classes=16, hidden=64, dropout=0.3, nhead=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.pos = PositionalEncoding(hidden, dropout)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, nhead, hidden * 4, dropout, batch_first=True),
            num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, n_classes)
    def forward(self, x):
        x = self.proj(x.transpose(1, 2))
        x = self.pos(x)
        x = self.enc(x)
        x = self.pool(x.transpose(1, 2)).flatten(1)
        return self.head(x)

def build_dl_model(name, n_classes=len(CLASSES), hidden=DL_HIDDEN, dropout=DL_DROPOUT):
    if name == "resnet":
        return ResNet1D(in_channels=15, n_classes=n_classes, hidden=hidden, dropout=dropout)
    if name == "gru":
        return GRUModel(in_channels=15, n_classes=n_classes, hidden=hidden, dropout=dropout)
    if name == "transformer":
        return TransformerModel(in_channels=15, n_classes=n_classes, hidden=hidden, dropout=dropout)
    raise ValueError(name)

def standardize(Xtr, Xte):
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True)
    return (Xtr - mean) / (std + 1e-8), (Xte - mean) / (std + 1e-8)

def predict(model, X, batch_size=DL_BATCH_SIZE, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            preds.append(logits.argmax(1))
        pred_tensor = torch.cat(preds)
    return pred_tensor.cpu().numpy()

def dl_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

def train_fold(model, Xtr, ytr, Xva, yva, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
               lr=DL_LR, weight_decay=DL_WEIGHT_DECAY, device="cpu", patience=6):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=DL_LR_STEP, gamma=DL_LR_GAMMA)
    lossf = nn.CrossEntropyLoss()
    n = len(ytr)

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)

    history = []
    best_val_acc = 0.0
    wait = 0
    best_weights = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            tr_pred = model(Xtr_t).argmax(1)
            val_pred = model(Xva_t).argmax(1)
            train_acc = (tr_pred == ytr_t).float().mean().item()
            val_acc = (val_pred == yva_t).float().mean().item()

        history.append({
            "epoch": len(history) + 1,
            "lr": sched.get_last_lr()[0],
            "train_loss": float(np.mean(losses)),
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
        sched.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_weights = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return history

def run_dl(model_name, X, y, meta, splits, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
           lr=DL_LR, weight_decay=DL_WEIGHT_DECAY, dropout=DL_DROPOUT, hidden=DL_HIDDEN,
           device="cpu", seed=42):
    set_seed(seed)
    fold_rows, yt_all, yp_all, histories = [], [], [], []
    for sp in splits:
        tr = meta["person_id"].isin(sp["train"]).to_numpy()
        te = meta["person_id"].isin(sp["test"]).to_numpy()
        Xtr_raw, Xte_raw = X[tr], X[te]
        ytr_raw, yte = y[tr], y[te]
        groups_tr = meta.loc[tr, "person_id"].to_numpy()
        # 被试不相交的内层验证划分：早停/选点只看内层验证集，杜绝外层测试集泄漏
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        itr_idx, iva_idx = next(gss.split(Xtr_raw, ytr_raw, groups=groups_tr))
        Xtr_i, ytr = Xtr_raw[itr_idx], ytr_raw[itr_idx]
        Xva_i, yva = Xtr_raw[iva_idx], ytr_raw[iva_idx]
        # 仅用内层训练集统计量标准化；内层验证集与外层测试集均不参与统计量估计
        _m = Xtr_i.mean(axis=(0, 2), keepdims=True)
        _s = Xtr_i.std(axis=(0, 2), keepdims=True)
        Xtr = (Xtr_i - _m) / (_s + 1e-8)
        Xva = (Xva_i - _m) / (_s + 1e-8)
        Xte = (Xte_raw - _m) / (_s + 1e-8)
        model = build_dl_model(model_name, hidden=hidden, dropout=dropout)
        history = train_fold(model, Xtr, ytr, Xva, yva, epochs=epochs, batch_size=batch_size,
                             lr=lr, weight_decay=weight_decay, device=device)
        pred = predict(model, Xte, batch_size, device)  # 仅在训练完成后对外层测试集评估一次
        fold_rows.append({"fold": sp["fold"], **dl_metrics(yte, pred)})
        yt_all.append(yte)
        yp_all.append(pred)
        hdf = pd.DataFrame(history)
        hdf["fold"] = sp["fold"]
        histories.append(hdf)
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    agg = dl_metrics(yt, yp)
    cm = confusion_matrix(yt, yp, labels=range(len(CLASSES)))
    return agg, cm, pd.DataFrame(fold_rows), pd.concat(histories, ignore_index=True)

# ===================== 主训练 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
dl_results = {}
for m in ["resnet", "gru", "transformer"]:
    agg, cm, folds, hist = run_dl(m, X, y, meta, splits, device=device)
    dl_results[m] = agg
    np.save(RESULTS_DIR / f"dl_{m}_confusion.npy", cm)
    folds.to_csv(RESULTS_TABLES_DIR / f"dl_{m}_folds.csv", index=False)
    hist.to_csv(RESULTS_TABLES_DIR / f"dl_{m}_history.csv", index=False)
    avg = hist.groupby("epoch")[["lr", "train_loss", "train_acc", "val_acc"]].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(avg["epoch"], avg["lr"]); axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_title("Learning rate")
    axes[0, 1].plot(avg["epoch"], avg["train_loss"]); axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_title("Training loss")
    axes[1, 0].plot(avg["epoch"], avg["train_acc"]); axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_title("Training accuracy"); axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(avg["epoch"], avg["val_acc"]); axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_title("Validation accuracy"); axes[1, 1].set_ylim(0, 1)
    fig.suptitle(m.upper()); fig.tight_layout()
    fig.savefig(RESULTS_FIGURES_DIR / f"dl_{m}_curves.png", dpi=150)
    # plt.show() # 正式跑实验建议注释，节省时间
    print(f"{m}: precision={agg['precision']:.4f} recall={agg['recall']:.4f} f1={agg['f1']:.4f} "
          f"weighted_f1={agg['weighted_f1']:.4f} accuracy={agg['accuracy']:.4f}")

# dropout消融实验（仅fold0，定性分析）
sp0 = splits[0]
tr0 = meta["person_id"].isin(sp0["train"]).to_numpy()
te0 = meta["person_id"].isin(sp0["test"]).to_numpy()
Xtr0_raw = X[tr0]
ytr0_raw = y[tr0]
groups0 = meta.loc[tr0, "person_id"].to_numpy()
gss0 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
i0, v0 = next(gss0.split(Xtr0_raw, ytr0_raw, groups=groups0))
Xtr0_i = Xtr0_raw[i0]
ytr0_i = ytr0_raw[i0]
Xva0_i = Xtr0_raw[v0]
yva0 = ytr0_raw[v0]
_m0 = Xtr0_i.mean(axis=(0, 2), keepdims=True)
_s0 = Xtr0_i.std(axis=(0, 2), keepdims=True)
Xtr0 = (Xtr0_i - _m0) / (_s0 + 1e-8)
Xva0 = (Xva0_i - _m0) / (_s0 + 1e-8)
drop_hists = {}
for lbl, drop in [("without dropout", 0.0), ("with dropout", DL_DROPOUT)]:
    set_seed(42)
    h = train_fold(build_dl_model("resnet", dropout=drop), Xtr0, ytr0_i, Xva0, yva0, device=device)
    drop_hists[lbl] = pd.DataFrame(h)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for lbl, h in drop_hists.items():
    axes[0, 0].plot(h["epoch"], h["lr"], label=lbl)
    axes[0, 1].plot(h["epoch"], h["train_loss"], label=lbl)
    axes[1, 0].plot(h["epoch"], h["train_acc"], label=lbl)
    axes[1, 1].plot(h["epoch"], h["val_acc"], label=lbl)
axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_title("Learning rate")
axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_title("Training loss")
axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_title("Training accuracy"); axes[1, 0].set_ylim(0, 1)
axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_title("Validation accuracy"); axes[1, 1].set_ylim(0, 1)
for ax in axes.ravel():
    ax.legend()
fig.suptitle("ResNet: effect of dropout (fold 0)")
fig.tight_layout()
fig.savefig(RESULTS_FIGURES_DIR / "dl_dropout_comparison.png", dpi=150)
# plt.show()

# ========== DL模型汇总：k折交叉验证 mean ± std（论文主表） ==========
summary_dl = []
for m_name in ["resnet","gru","transformer"]:
    df_fold = pd.read_csv(RESULTS_TABLES_DIR / f"dl_{m_name}_folds.csv")
    stat = df_fold[["precision","recall","f1","weighted_f1","accuracy"]].agg(["mean","std"])
    summary_dl.append({
        "model": m_name,
        "accuracy_mean": stat.loc["mean","accuracy"],
        "accuracy_std": stat.loc["std","accuracy"],
        "mp_mean": stat.loc["mean","precision"],
        "mp_std": stat.loc["std","precision"],
        "mr_mean": stat.loc["mean","recall"],
        "mr_std": stat.loc["std","recall"],
        "wf1_mean": stat.loc["mean","weighted_f1"],
        "wf1_std": stat.loc["std","weighted_f1"],
        "mf1_mean": stat.loc["mean","f1"],
        "mf1_std": stat.loc["std","f1"],
    })
summary_df_dl = pd.DataFrame(summary_dl)
summary_df_dl.to_csv(RESULTS_TABLES_DIR / "dl_all_model_summary_mean_std.csv", index=False)
print("\n==== DL模型汇总（k折交叉验证 mean ± std） ====")
print(summary_df_dl.round(4))

ABLATION_SETS = {
    "仅双腕(6)": SENSOR_GROUPS["wrists"],
    "仅双踝(6)": SENSOR_GROUPS["ankles"],
    "仅后腰(3)": SENSOR_GROUPS["waist"],
    "上肢(双腕+后腰,9)": SENSOR_GROUPS["wrists"] + SENSOR_GROUPS["waist"],
    "下肢(双踝+后腰,9)": SENSOR_GROUPS["ankles"] + SENSOR_GROUPS["waist"],
    "全部(15)": CHANNELS,
}
rows = []
for name, chans in ABLATION_SETS.items():
    idx = [CHANNELS.index(c) for c in chans]
    Fsub = extract_features(X[:, idx, :])
    # 6个返回值全部接收
    agg, cm, df_folds, df_global_perclass, df_fold_perclass, df_bestparams = run_cv("svm", SVM_PARAM_GRID, Fsub, y, meta, splits)
    rows.append({
        "sensor_set": name,
        "n_channels": len(chans),
        "n_features": Fsub.shape[1],
        "macro_f1": round(agg["macro_f1"], 4),
        "weighted_f1": round(agg["weighted_f1"], 4),
        "accuracy": round(agg["accuracy"], 4),
    })
ablation = pd.DataFrame(rows)
ablation.to_csv(RESULTS_TABLES_DIR / "sensor_ablation.csv", index=False)
display(ablation)

def plot_confusion(cm_path, title="Confusion matrix"):
    cm = np.load(cm_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制混淆矩阵
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    # 坐标轴
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=90)
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # ========================================================
    # 新增：在混淆矩阵每个格子中显示数字
    # ========================================================
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8
            )
    # ========================================================

    fig.tight_layout()

    fig.savefig(
        RESULTS_FIGURES_DIR / "confusion_heatmap_svm.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


# 原来的调用方式保持不变
if (RESULTS_DIR / "ml_svm_confusion.npy").exists():
    plot_confusion(RESULTS_DIR / "ml_svm_confusion.npy")

PAIRS = [
    ("Left vs Right Roundhouse Kick",
     {"Left Roundhouse Kick": ["Left Roundhouse Kick"],
      "Right Roundhouse Kick": ["Right Roundhouse Kick"]}),
    ("Front vs Side Kick",
     {"Front Kick": ["Left Front Kick", "Right Front Kick"],
      "Side Kick": ["Left Side Kick", "Right Side Kick"]}),
    ("Straight Punch vs Block",
     {"Straight Punch": ["Left Straight Punch", "Right Straight Punch"],
      "Block": ["Left High Block", "Right High Block", "Left Middle Block",
                "Right Middle Block", "Left Low Block", "Right Low Block"]}),
]


def mean_waveform(X, idx):
    w = np.mean([resultant_magnitude(X[i]) for i in idx], axis=0)
    return w - w.mean()


fig, axes = plt.subplots(1, 3, figsize=(16, 4))
t = np.arange(120) / FS
for ax, (title, groups) in zip(axes, PAIRS):
    for label, actions in groups.items():
        idx = [i for i, a in enumerate(meta["action"]) if a in actions]
        ax.plot(t, mean_waveform(X, idx), label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(title)
    ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_FIGURES_DIR / "waveform_comparison.png", dpi=150)
plt.show()

spec = group_spectra(meta, X)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (gname, (f, ps, frac, n)) in zip(axes.ravel(), spec.items()):
    ax.plot(f, ps)
    ax.axvline(15.0, color="r", ls="--", label="15 Hz")
    ax.set_title(f"{gname} (n={n})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.legend()
fig.suptitle("Mean power spectrum per kick group")
fig.tight_layout()
fig.savefig(RESULTS_FIGURES_DIR / "spectrum_overview.png", dpi=150)
plt.show()


def plot_paper_waveform(X, meta, out_path):
    """论文图 13/14：右脚腕 y/z 轴，右前踢 vs 右横踢。"""
    actions = ["Right Front Kick", "Right Roundhouse Kick"]
    chans = {"Y axis (y_rL)": "y_rL", "Z axis (z_rL)": "z_rL"}
    t = np.arange(120) / FS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (title, c) in zip(axes, chans.items()):
        ci = CHANNELS.index(c)
        for a in actions:
            idx = [i for i, act in enumerate(meta["action"]) if act == a]
            w = np.mean([X[i][ci] for i in idx], axis=0)
            ax.plot(t, w, label=a)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (g)")
        ax.set_title(f"Right ankle {title}")
        ax.legend()
    fig.suptitle("Right front kick vs right roundhouse kick (right ankle)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_paper_waveform(X, meta, RESULTS_FIGURES_DIR / "paper_waveform_right_kick.png")

# ========== 正确分类：按动作类型拆分，不是按左右拆分 ==========
KICK_ACTIONS = [
    "Left Back Kick",
    "Left Front Kick",
    "Left Roundhouse Kick",
    "Left Side Kick",
    "Right Back Kick",
    "Right Front Kick",
    "Right Roundhouse Kick",
    "Right Side Kick",
]

PUNCH_ACTIONS = [
    "Left High Block",
    "Left Low Block",
    "Left Middle Block",
    "Left Straight Punch",
    "Right High Block",
    "Right Low Block",
    "Right Middle Block",
    "Right Straight Punch",
]

def _subject_sort_key(key):
    region, sid = key.split("_", 1)
    return (region, int(sid))

def region_table(df, region, actions):
    sub = df[df["region"] == region]
    rows = []
    for sk in sorted(sub["subject_key"].unique(), key=_subject_sort_key):
        sk_data = sub[sub["subject_key"] == sk].drop_duplicates("id")
        cnt = sk_data.groupby("action").size()
        row = {"subject_key": sk, "total": int(cnt.sum())}
        for a in actions:
            row[a] = int(cnt.get(a, 0))
        rows.append(row)
    return pd.DataFrame(rows)

# ========== 生成两张统计表 ==========
# kick组受试者 → 统计8个踢击动作
table_kick = region_table(df, "kick", KICK_ACTIONS)
table_kick.to_csv(RESULTS_TABLES_DIR / "metadata_kick.csv", index=False)
print(f"--- kick组 踢击动作 --- 缺失(0次)单元数：{(table_kick[KICK_ACTIONS] == 0).sum().sum()}")
display(table_kick)

# punch组受试者 → 统计8个格挡+拳击动作
table_punch = region_table(df, "punch", PUNCH_ACTIONS)
table_punch.to_csv(RESULTS_TABLES_DIR / "metadata_punch.csv", index=False)
print(f"\n--- punch组 格挡拳击动作 --- 缺失(0次)单元数：{(table_punch[PUNCH_ACTIONS] == 0).sum().sum()}")
display(table_punch)

import numpy as np

# 方案B：遍历splits，直接从meta查看每折测试集包含的动作类别
all_test_labels = []
for sp in splits:
    tr_subjects = sp["train"]
    te_subjects = sp["test"]
    te_mask = meta["person_id"].isin(te_subjects)
    test_act = meta.loc[te_mask, "action"].values
    all_test_labels.extend(test_act)
    print(f"折 {sp['fold']} 测试集动作类别：{np.unique(test_act)}")

all_test_labels = np.array(all_test_labels)
print("\n===== 全部折合并后的测试集动作 =====")
print("所有测试集出现过的动作标签：", np.unique(all_test_labels))
uniq, cnt = np.unique(all_test_labels, return_counts=True)
for act, num in zip(uniq, cnt):
    print(f"{act} 测试样本数量：{num}")
