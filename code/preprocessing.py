#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing for the five-body-site Taekwondo accelerometer dataset.

Converts the source workbooks (two of which are CSV text saved with an .xlsx
extension and two of which are genuine Excel files) into standard UTF-8 CSV,
translates the kicking action names to English for consistency, and runs
integrity / range checks.

Usage
-----
    python preprocessing.py \
        --src "E:/yanjiusheng/datas/腿法/dataset" \
        --dst "../data"

Paths can also be supplied through the environment variables
TAEKWONDO_SOURCE_ROOT and TAEKWONDO_DATA_ROOT.

Dependencies: pandas, numpy, openpyxl  (see requirements.txt)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
KICK2EN = {
    "左前踢": "Left Front Kick",
    "右前踢": "Right Front Kick",
    "左横踢": "Left Roundhouse Kick",
    "右横踢": "Right Roundhouse Kick",
    "左侧踢": "Left Side Kick",
    "右侧踢": "Right Side Kick",
    "左后踢": "Left Back Kick",
    "右后踢": "Right Back Kick",
}

CHANNELS = [
    "x_W", "y_W", "z_W",
    "x_rL", "y_rL", "z_rL",
    "x_rH", "y_rH", "z_rH",
    "x_lL", "y_lL", "z_lL",
    "x_lH", "y_lH", "z_lH",
]

POINTS_PER_WINDOW = 120
SENSOR_RANGE_G = 6.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_table(path: Path) -> pd.DataFrame:
    """Read a source table that is either a real .xlsx file or CSV text."""
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def translate_kick_id(id_str: str) -> str:
    action, subject, trial = str(id_str).strip().rsplit("_", 2)
    return f"{KICK2EN[action]}_{subject}_{trial}"


def translate_kick_label(label: str) -> str:
    return KICK2EN[str(label).strip()]


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_region(src_dir: Path, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if region == "kick":
        data_path = src_dir / "kick technique" / "data_kick_technique.xlsx"
        label_path = src_dir / "kick technique" / "label_kick technique.xlsx"
    else:
        data_path = src_dir / "punch technique" / "data_punch technique.xlsx"
        label_path = src_dir / "punch technique" / "label_punch_technique.xlsx"

    data = read_table(data_path)
    label = read_table(label_path)
    data["id"] = data["id"].astype(str).str.strip()
    label["id"] = label["id"].astype(str).str.strip()

    if region == "kick":
        data["id"] = data["id"].map(translate_kick_id)
        label["id"] = label["id"].map(translate_kick_id)
        label["label"] = label["label"].map(translate_kick_label)

    return data, label


# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
def check_region(data: pd.DataFrame, label: pd.DataFrame, region: str) -> None:
    # 1. id consistency between data and label files
    data_ids = set(data["id"].unique())
    label_ids = set(label["id"].unique())
    assert data_ids == label_ids, f"{region}: data/label id sets differ"

    # 2. exactly 120 points per id, time index 0..119
    counts = data.groupby("id")["time"].count()
    assert (counts == POINTS_PER_WINDOW).all(), f"{region}: windows != 120 points"
    tmin = data.groupby("id")["time"].min()
    tmax = data.groupby("id")["time"].max()
    assert (tmin == 0).all() and (tmax == POINTS_PER_WINDOW - 1).all(), \
        f"{region}: time index not 0..119"

    # 3. no missing values
    assert not data[CHANNELS].isna().any().any(), f"{region}: NaN in signal channels"

    # 4. range check (report only; transient exceedances are retained)
    arr = data[CHANNELS].to_numpy()
    over = int((np.abs(arr) >= SENSOR_RANGE_G).sum())
    print(f"[{region}] windows={len(label_ids)} rows={len(data)} "
          f"points/window ok, no NaN, |a|>=6 g readings={over}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=os.environ.get(
        "TAEKWONDO_SOURCE_ROOT", r"E:\yanjiusheng\datas\腿法\dataset"),
        help="Source dataset root containing kick/punch technique folders")
    parser.add_argument("--dst", default=os.environ.get(
        "TAEKWONDO_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data")),
        help="Output directory for the standard CSV files")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "kick": ("data_kick_technique.csv", "label_kick_technique.csv"),
        "punch": ("data_punch_technique.csv", "label_punch_technique.csv"),
    }

    for region, (data_name, label_name) in outputs.items():
        data, label = convert_region(src_dir, region)
        check_region(data, label, region)
        data.to_csv(dst_dir / data_name, index=False, encoding="utf-8")
        label.to_csv(dst_dir / label_name, index=False, encoding="utf-8")
        print(f"[{region}] wrote {data_name} and {label_name} -> {dst_dir}")

    print("Preprocessing and integrity checks completed successfully.")


if __name__ == "__main__":
    main()
