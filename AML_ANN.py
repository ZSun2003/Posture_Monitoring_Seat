# -*- coding: utf-8 -*-
"""
AML_ANN.py  (ANN/MLP 版本, Updated: ToF 门控在采集端完成)

====================================================================
本脚本用途(PC 端训练 + 导出嵌入式可用模型):
--------------------------------------------------------------------
1) 从坐姿 CSV/Excel(导出为 CSV) 数据中读取并整理"工程特征"(不额外生成新列),作为 ANN(MLP) 的输入.
2) 在 PC 上训练 ANN 分类器(多层感知机 MLP, PyTorch 实现):
   - 可配置隐藏层层数/每层神经元数、激活函数(relu/tanh/sigmoid)、dropout、优化器超参数等
   - 使用 StandardScaler 做输入标准化(与嵌入式推理保持一致)
3) 在测试集上评估(accuracy / classification report / confusion matrix 等).
4) 可选导出为 Arduino/ESP32 可直接编译使用的 C 头文件:
   - 默认文件名: AML_ANN.h (内含 ANN 权重/偏置与 StandardScaler 参数)
   - 如需更名,可在全局变量区修改 OUT_HEADER_NAME

====================================================================
数据结构:
--------------------------------------------------------------------
AMLDEVICE/
  Straight/participant_1.csv
  Left/participant_1.csv
  ...
====================================================================
"""

from __future__ import annotations

import sys
import argparse
import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# ★★ 所有可调参数(请只在这里修改默认值) ★★
# ============================================================

# ---------- 运行系统选择(只在这里改) ----------
AML_OS = "MAC"  # "WIN" / "MAC"

if AML_OS.strip().upper() == "WIN":
    DATA_ROOT_PATH = r"C:\Users\17717\Desktop\AML_NEW_DATA"
    OUTPUT_ROOT_PATH = r"C:\Users\17717\Desktop\AML_MODEL\ANN"
elif AML_OS.strip().upper() == "MAC":
    _desktop = Path.home() / "Desktop"
    DATA_ROOT_PATH = str(_desktop / "AML_CODE"/ "AML_DATASET")
    OUTPUT_ROOT_PATH = str(_desktop / "AML_MODEL" / "ANN")
else:
    raise ValueError("AML_OS 只能是 'WIN' 或 'MAC'.")

# ---------- 固定输出文件名 ----------
OUT_HEADER_NAME = "AML_ANN.h"
OUT_CONFIG_NAME = "AML_ANN_config.json"
OUT_CM_NAME = "AML_ANN_CM.jpg"
OUT_BUNDLE_NAME = "AML_ANN_bundle.joblib"

OUT_CV_SUMMARY_NAME = "AML_ANN_CV_summary.json"
OUT_CV_CM_NAME = "AML_ANN_CV_CM.jpg"

# ---------- 训练后行为开关 ----------
ENABLE_MODEL_EXPORT = False

# ---------- Cross Validation ----------
ENABLE_CROSS_VALIDATION = False
CROSS_VALIDATION_METHOD = "LeaveOneGroupOut"  # "GroupKFold" / "LeaveOneGroupOut"
CROSS_VALIDATION_N_SPLITS = 5

# ---------- 默认训练数据采样参数 ----------
DEFAULT_MAX_FILES_PER_POSTURE = 0
DEFAULT_ROWS_PER_POSTURE = 0
DEFAULT_ROWS_PER_FILE = 0
DEFAULT_SEED = 42
DEFAULT_NO_CONFIRM = False

# ---------- 输入特征选择 ----------
TRAIN_FEATURE_KEYS = ["FSR", "TOF", "lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio"]

# ---------- Feature Dropout / Sensor Masking ----------
ENABLE_FSR_SENSOR_MASKING = True
FSR_SENSOR_MASK_PROB = 0.0  # 建议 0.05~0.30

# ---------- Domain Randomization (FSR) ----------
ENABLE_FSR_DOMAIN_RANDOMIZATION = True
FSR_DR_SCALE_RANGE = (0.8, 1.2)
FSR_DR_NOISE_STD = 0.05
FSR_DR_BIAS_STD = 0.0

# ---------- Feature Weighting (与 SVM 一致: StandardScaler 之后乘权重) ----------
ENABLE_FEATURE_WEIGHTING = True
TOF_GATED_WEIGHT = 2.0
TOF_USE_WEIGHT = 1.0  # 默认不加权 tof_use

# ---------- group 解析规则 ----------
DEFAULT_PARTICIPANT_REGEX = r"(participant[_-]?\d+)"
DEFAULT_GROUP_LOWERCASE = True

# ---------- 数据集划分 ----------
TRAIN_SET_PARTICIPANT_NUMBERS = [1, 2, 3, 5, 6]
VAL_SET_PARTICIPANT_NUMBERS = [7]
TEST_SET_PARTICIPANT_NUMBERS = [4]

# ---------- 默认模型参数(ANN / MLP) ----------
DEFAULT_ANN_HIDDEN_LAYERS = [256, 256]
DEFAULT_ANN_ACTIVATION = "relu"  # "relu" / "tanh" / "sigmoid"
DEFAULT_ANN_DROPOUT = 0.05
DEFAULT_ANN_BATCH_SIZE = 64
DEFAULT_ANN_EPOCHS = 300
DEFAULT_ANN_LR = 1e-3
DEFAULT_ANN_WEIGHT_DECAY = 1e-4

# ---- 学习率动态调整 ----
DEFAULT_ANN_LR_SCHEDULER = "plateau"  # "none" / "plateau"
DEFAULT_ANN_LR_MIN = 5e-6
DEFAULT_ANN_LR_PLATEAU_FACTOR = 0.5
DEFAULT_ANN_LR_PLATEAU_PATIENCE = 10
DEFAULT_ANN_LR_PLATEAU_THRESHOLD = 1e-5
DEFAULT_ANN_LR_PLATEAU_COOLDOWN = 0

# ---- weight_decay 动态调整 ----
DEFAULT_ANN_WD_SCHEDULER = "none"  # "none" / "linear"
DEFAULT_ANN_WEIGHT_DECAY_END = 0.0

DEFAULT_ANN_EARLY_STOP_PATIENCE = 20
DEFAULT_ANN_MIN_DELTA = 0.0
DEFAULT_ANN_USE_CLASS_WEIGHTS = True

DEFAULT_ANN_DEVICE = "GPU"  # "CPU" / "GPU"
DEFAULT_ANN_VERBOSE = 1

# ---------- 禁用的命令行参数 ----------
FORBIDDEN_CLI_ARGS = ("--data_root", "--output_dir")

# ============================================================
# 1) 特征列名与顺序
# ============================================================

FSR_FEATURE_NAMES: List[str] = [f"fsr{i}_norm" for i in range(10)]

FSR_DERIVED_FEATURE_NAMES: List[str] = [
    "lr_ratio",
    "ud_ratio",
    "cop_x_ratio",
    "cop_y_ratio",
]

TOF_FEATURE_NAMES: List[str] = [
    "tof_gated",
    "tof_use",
]

FEATURE_NAMES_WITH_TOF: List[str] = (
    FSR_FEATURE_NAMES
    + TOF_FEATURE_NAMES
    + FSR_DERIVED_FEATURE_NAMES
)

FEATURE_NAMES_NO_TOF: List[str] = (
    FSR_FEATURE_NAMES
    + FSR_DERIVED_FEATURE_NAMES
)

_FEATURE_KEYS_CANONICAL: List[str] = [
    "FSR",
    "TOF",
    "lr_ratio",
    "ud_ratio",
    "cop_x_ratio",
    "cop_y_ratio",
]
_ALLOWED_FEATURE_KEYS: set[str] = set(_FEATURE_KEYS_CANONICAL) | {"ALL"}


def _as_str_list(v: object) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        parts = re.split(r"[\s,;]+", s)
        return [p for p in (x.strip() for x in parts) if p]
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return [str(x).strip() for x in list(v) if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def _normalize_feature_key(k: str) -> str:
    kk = str(k).strip()
    if not kk:
        return ""
    up = kk.upper()
    if up in ("FSR", "TOF", "ALL"):
        return up
    low = kk.lower()
    if low in FSR_DERIVED_FEATURE_NAMES:
        return low
    return kk


def resolve_model_feature_names(feature_keys: object) -> List[str]:
    keys_raw = _as_str_list(feature_keys)
    keys = [_normalize_feature_key(k) for k in keys_raw if _normalize_feature_key(k)]

    if any(k == "ALL" for k in keys):
        keys = list(_FEATURE_KEYS_CANONICAL)

    unknown = sorted({k for k in keys if k not in _ALLOWED_FEATURE_KEYS})
    if unknown:
        raise ValueError(
            "TRAIN_FEATURE_KEYS 包含不支持的 feature key: "
            f"{unknown}\n"
            f"允许的 key: {sorted(_ALLOWED_FEATURE_KEYS)}\n"
            "示例: ['FSR','TOF'] 或 ['FSR','TOF','lr_ratio','ud_ratio','cop_x_ratio','cop_y_ratio']"
        )

    selected = set(keys)
    cols: List[str] = []

    if "FSR" in selected:
        cols.extend(FSR_FEATURE_NAMES)
    if "TOF" in selected:
        cols.extend(TOF_FEATURE_NAMES)
    for n in FSR_DERIVED_FEATURE_NAMES:
        if n in selected:
            cols.append(n)

    if not cols:
        raise ValueError("TRAIN_FEATURE_KEYS 解析后得到 0 维输入,无法训练.")

    seen: set[str] = set()
    dedup: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


# ============================================================
# 2) 小工具函数
# ============================================================

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        s = input(prompt + suffix).strip().lower()
        if s == "":
            return default
        if s in ("y", "yes", "是", "继续", "1"):
            return True
        if s in ("n", "no", "否", "停止", "0"):
            return False
        print("输入无效,请输入 y/yes 或 n/no(回车采用默认选项).")


def extract_participant_group_id(file_path: Path) -> str:
    stem = file_path.stem
    m = re.search(DEFAULT_PARTICIPANT_REGEX, stem, flags=re.IGNORECASE)
    gid = m.group(1) if m else stem
    return gid.lower() if DEFAULT_GROUP_LOWERCASE else gid


# ============================================================
# 3) 目录扫描
# ============================================================

def find_posture_sources_folder_only(data_root: Path, posture: str) -> List[Path]:
    posture_dir = data_root / posture
    if not posture_dir.exists() or not posture_dir.is_dir():
        return []
    return sorted(posture_dir.glob("*.csv"), key=lambda p: natural_sort_key(p.name))


def auto_detect_postures_folder_only(data_root: Path) -> List[str]:
    if not data_root.exists():
        return []
    postures: List[str] = []
    for p in data_root.iterdir():
        if p.is_dir() and any(p.glob("*.csv")):
            postures.append(p.name)
    return sorted(postures, key=natural_sort_key)


def scan_dataset_structure(
    data_root: Path,
    detected_postures: Sequence[str],
    selected_postures: Sequence[str],
    max_files_per_posture: Optional[int],
) -> Dict[str, Dict[str, int]]:
    info: Dict[str, Dict[str, int]] = {}
    selected_set = set(selected_postures)
    for posture in detected_postures:
        files = find_posture_sources_folder_only(data_root, posture)
        total_files = len(files)
        if posture in selected_set:
            used_files = total_files if max_files_per_posture is None else min(total_files, max_files_per_posture)
        else:
            used_files = 0
        info[posture] = {"total_files": total_files, "used_files": used_files}
    return info


# ============================================================
# 4) 读取 CSV 并构造特征
# ============================================================

@dataclass
class FeatureBuildConfig:
    feature_keys: List[str] = field(default_factory=lambda: list(TRAIN_FEATURE_KEYS))


def _require_cols(df: pd.DataFrame, cols: List[str], file_path: Path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列:{missing}\n"
            f"文件:{file_path}\n"
            f"现有列:{list(df.columns)}"
        )


def get_model_feature_names(feature_keys: object) -> List[str]:
    return resolve_model_feature_names(feature_keys)


def build_features_from_df(df: pd.DataFrame, cfg: FeatureBuildConfig, file_path: Path) -> np.ndarray:
    model_features = get_model_feature_names(cfg.feature_keys)
    _require_cols(df, model_features, file_path)
    X = df[model_features].to_numpy(dtype=np.float32)
    if X.shape[0] == 0:
        return np.zeros((0, len(model_features)), dtype=np.float32)
    if X.shape[1] != len(model_features):
        raise RuntimeError(f"内部错误:特征维度不是 {len(model_features)},而是 {X.shape[1]}")
    return X.astype(np.float32)


def load_one_excel(file_path: Path, feat_cfg: FeatureBuildConfig) -> np.ndarray:
    model_features = get_model_feature_names(feat_cfg.feature_keys)
    wanted_cols = set(
        model_features
        + ["tof_norm", "t_ms"]
        + [f"fsr{i}_std" for i in range(10)]
    )

    header_cols = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        sep=",",
        nrows=0,
    ).columns.tolist()

    usecols = [c for c in header_cols if c in wanted_cols]
    if not usecols:
        raise ValueError(f"CSV 文件没有任何需要的列,无法读取特征. 文件:{file_path}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        sep=",",
        engine="c",
        usecols=usecols,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    if df.shape[0] == 0:
        return np.zeros((0, len(model_features)), dtype=np.float32)

    X = build_features_from_df(df, feat_cfg, file_path)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if X.shape[0] == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    fsr_idx = [i for i, n in enumerate(model_features) if n in set(FSR_FEATURE_NAMES)]
    if fsr_idx:
        fsr_sum = np.sum(X[:, fsr_idx], axis=1)
        X = X[fsr_sum > 1e-6]

    return X.astype(np.float32)


def build_dataset(
    data_root: Path,
    postures: Sequence[str],
    max_files_per_posture: Optional[int],
    rows_per_posture: Optional[int],
    rows_per_file: Optional[int],
    seed: int,
    feat_cfg: FeatureBuildConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    g_list: List[str] = []

    for posture in postures:
        files = find_posture_sources_folder_only(data_root, posture)
        if not files:
            raise FileNotFoundError(
                f"未找到坐姿 {posture} 的数据文件.\n"
                f"请检查:{data_root / posture} 下是否包含 participant_*.csv"
            )

        if max_files_per_posture is not None and len(files) > max_files_per_posture:
            pick = rng.permutation(len(files))[:max_files_per_posture]
            files = [files[i] for i in pick]

        posture_rows: List[np.ndarray] = []
        posture_groups: List[str] = []

        for f in files:
            Xf = load_one_excel(f, feat_cfg)
            if Xf.size == 0:
                continue
            if rows_per_file is not None and len(Xf) > rows_per_file:
                idx = rng.choice(len(Xf), size=rows_per_file, replace=False)
                Xf = Xf[idx]
            posture_rows.append(Xf)
            posture_groups.append(extract_participant_group_id(f))

        if not posture_rows:
            raise RuntimeError(f"坐姿 {posture} 的所有文件读取后为空,请检查数据.")

        Xp = np.vstack(posture_rows)
        gp = np.concatenate(
            [np.full(len(Xf), posture_groups[i], dtype=object) for i, Xf in enumerate(posture_rows)]
        )

        if rows_per_posture is not None and len(Xp) > rows_per_posture:
            idx = rng.choice(len(Xp), size=rows_per_posture, replace=False)
            Xp = Xp[idx]
            gp = gp[idx]

        if len(Xp) > 1:
            perm = rng.permutation(len(Xp))
            Xp = Xp[perm]
            gp = gp[perm]

        X_list.append(Xp)
        y_list.extend([posture] * len(Xp))
        g_list.extend(gp.tolist())

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.array(y_list, dtype=object)
    g_all = np.array(g_list, dtype=object)
    return X_all, y_all, g_all


def _build_posture_file_map(files: Sequence[Path]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for f in files:
        gid = extract_participant_group_id(f)
        if gid in m:
            raise RuntimeError(
                f"检测到重复 participant(group_id)={gid} 的文件:\n"
                f"  1) {m[gid]}\n"
                f"  2) {f}\n"
                f"请检查文件命名,确保每个 posture 内 participant_*.csv 唯一."
            )
        m[gid] = f
    return m


def build_dataset_splits(
    data_root: Path,
    postures: Sequence[str],
    max_files_per_posture: Optional[int],
    rows_per_posture: Optional[int],
    rows_per_file: Optional[int],
    seed: int,
    feat_cfg: FeatureBuildConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    str, Dict[str, object],
]:
    rng = np.random.default_rng(seed)
    feat_dim = int(len(get_model_feature_names(feat_cfg.feature_keys)))

    if not postures:
        raise ValueError("postures 为空,无法构建数据集.")

    base_posture = "Straight" if ("Straight" in set(postures)) else str(postures[0])

    posture_maps: Dict[str, Dict[str, Path]] = {}
    for posture in postures:
        files = find_posture_sources_folder_only(data_root, posture)
        if not files:
            raise FileNotFoundError(
                f"未找到坐姿 {posture} 的数据文件.\n"
                f"请检查:{data_root / posture} 下是否包含 participant_*.csv"
            )
        posture_maps[posture] = _build_posture_file_map(files)

    if base_posture not in posture_maps:
        raise RuntimeError(f"基准 posture={base_posture} 不存在于当前 postures 列表中,无法划分 participant.")

    base_ids_all = sorted(list(posture_maps[base_posture].keys()), key=natural_sort_key)
    if not base_ids_all:
        raise RuntimeError(f"基准 posture={base_posture} 下没有任何 participant 文件,无法训练.")

    base_cnt = len(base_ids_all)
    for posture in postures:
        cnt = len(posture_maps[posture])
        if cnt != base_cnt:
            raise RuntimeError(
                f"participant 数量不一致: posture={posture} 有 {cnt} 个, "
                f"但基准 posture={base_posture} 有 {base_cnt} 个.\n"
                f"请保证每个 posture 的 participant 文件数量一致."
            )

    participant_total = int(len(base_ids_all))
    split_mode = "sample_ratio_per_posture" if participant_total < 3 else "participant_manual"

    selected_ids: List[str] = []
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    if split_mode == "participant_manual":
        num_to_gid: Dict[int, str] = {}
        for gid in base_ids_all:
            m = re.fullmatch(r"participant[_-]?(\d+)", str(gid), flags=re.IGNORECASE)
            if not m:
                continue
            num = int(m.group(1))
            if (num in num_to_gid) and (num_to_gid[num] != str(gid)):
                raise RuntimeError(
                    "检测到 participant 编号重复(同一个数字对应多个 group_id),无法使用手动划分.\n"
                    f"num={num} -> {num_to_gid[num]} / {gid}"
                )
            num_to_gid[num] = str(gid)

        if not num_to_gid:
            raise RuntimeError(
                "无法从 participant 文件名中提取数字编号(期望形如 participant_1.csv).\n"
                f"当前基准 posture={base_posture} 的 group_id 示例: {base_ids_all[:5]}\n"
                "请检查文件命名是否包含 participant_<数字>."
            )

        def _as_int_list(v: object) -> List[int]:
            if v is None:
                return []
            if isinstance(v, bool):
                return []
            if isinstance(v, int):
                return [int(v)]
            if isinstance(v, (list, tuple, set, np.ndarray)):
                return [int(x) for x in list(v)]
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                parts = re.split(r"[\s,;]+", s)
                return [int(p) for p in parts if p != ""]
            raise TypeError("TRAIN/VAL/TEST_SET_PARTICIPANT_NUMBERS 必须是 int 或 int 列表(例如 [1,2,3]).")

        tr_nums = _as_int_list(TRAIN_SET_PARTICIPANT_NUMBERS)
        val_nums = _as_int_list(VAL_SET_PARTICIPANT_NUMBERS)
        te_nums = _as_int_list(TEST_SET_PARTICIPANT_NUMBERS)

        all_nums = [int(n) for n in (tr_nums + val_nums + te_nums)]
        if any(int(n) <= 0 for n in all_nums):
            raise RuntimeError(f"participant 编号必须为正整数,当前: {all_nums}")

        missing = [int(n) for n in all_nums if int(n) not in num_to_gid]
        if missing:
            avail = sorted(num_to_gid.keys())
            raise RuntimeError(
                "你在全局变量区指定了不存在的 participant 编号,无法划分数据集.\n"
                f"缺失编号: {sorted(set(missing))}\n"
                f"可用编号: {avail}"
            )

        train_ids = set([num_to_gid[int(n)] for n in tr_nums])
        val_ids = set([num_to_gid[int(n)] for n in val_nums])
        test_ids = set([num_to_gid[int(n)] for n in te_nums])

        if len(train_ids) <= 0 or len(test_ids) <= 0:
            raise RuntimeError(
                "训练集/测试集 participant 数量不能为 0(否则不训练).\n"
                f"当前: train={sorted(set(tr_nums))}, test={sorted(set(te_nums))}"
            )

        if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
            raise RuntimeError(
                "train/val/test participant 编号不能重叠.\n"
                f"train∩val={sorted(train_ids & val_ids, key=natural_sort_key)}, "
                f"train∩test={sorted(train_ids & test_ids, key=natural_sort_key)}, "
                f"val∩test={sorted(val_ids & test_ids, key=natural_sort_key)}"
            )

        selected_ids = sorted(list(train_ids | val_ids | test_ids), key=natural_sort_key)

        if max_files_per_posture is not None:
            print("[INFO] 已启用手动 participant 划分:将忽略 max_files_per_posture 限制.")
    else:
        selected_ids = list(base_ids_all)
        if (max_files_per_posture is not None) and (len(selected_ids) > int(max_files_per_posture)):
            pick = rng.permutation(len(selected_ids))[: int(max_files_per_posture)]
            selected_ids = [selected_ids[i] for i in pick]
            selected_ids = sorted(selected_ids, key=natural_sort_key)

    participant_cnt = int(len(selected_ids))
    if participant_cnt <= 0:
        raise RuntimeError("可用 participant 数量为 0,无法训练.")

    X_tr_list: List[np.ndarray] = []
    y_tr_list: List[str] = []
    g_tr_list: List[str] = []

    X_val_list: List[np.ndarray] = []
    y_val_list: List[str] = []
    g_val_list: List[str] = []

    X_te_list: List[np.ndarray] = []
    y_te_list: List[str] = []
    g_te_list: List[str] = []

    for posture in postures:
        pmap = posture_maps[posture]
        for gid in selected_ids:
            if gid not in pmap:
                raise RuntimeError(f"posture={posture} 缺少 participant(group_id)={gid} 的文件,无法保证一致划分.")

        used_ids = list(selected_ids)
        posture_rows: List[np.ndarray] = []
        posture_groups: List[str] = []

        for gid in used_ids:
            f = pmap[gid]
            Xf = load_one_excel(f, feat_cfg)
            if Xf.size == 0:
                continue
            if rows_per_file is not None and len(Xf) > rows_per_file:
                idx = rng.choice(len(Xf), size=rows_per_file, replace=False)
                Xf = Xf[idx]
            posture_rows.append(Xf)
            posture_groups.append(gid)

        if not posture_rows:
            raise RuntimeError(f"坐姿 {posture} 的所有文件读取后为空,请检查数据.")

        Xp = np.vstack(posture_rows).astype(np.float32)
        gp = np.concatenate(
            [np.full(len(Xf), posture_groups[i], dtype=object) for i, Xf in enumerate(posture_rows)]
        )

        if rows_per_posture is not None and len(Xp) > rows_per_posture:
            idx = rng.choice(len(Xp), size=rows_per_posture, replace=False)
            Xp = Xp[idx]
            gp = gp[idx]

        if len(Xp) > 1:
            perm = rng.permutation(len(Xp))
            Xp = Xp[perm]
            gp = gp[perm]

        if split_mode == "sample_ratio_per_posture":
            n_total = int(len(Xp))
            n_val = int(n_total * 0.15)
            n_test = int(n_total * 0.15)
            n_train = int(n_total - n_val - n_test)

            Xp_tr = Xp[:n_train]
            gp_tr = gp[:n_train]

            Xp_val = Xp[n_train:n_train + n_val]
            gp_val = gp[n_train:n_train + n_val]

            Xp_te = Xp[n_train + n_val:n_train + n_val + n_test]
            gp_te = gp[n_train + n_val:n_train + n_val + n_test]
        else:
            mask_tr = np.isin(gp, list(train_ids))
            mask_val = np.isin(gp, list(val_ids)) if val_ids else np.zeros(len(gp), dtype=bool)
            mask_te = np.isin(gp, list(test_ids))

            Xp_tr, gp_tr = Xp[mask_tr], gp[mask_tr]
            Xp_val, gp_val = Xp[mask_val], gp[mask_val]
            Xp_te, gp_te = Xp[mask_te], gp[mask_te]

        if len(Xp_tr) > 0:
            X_tr_list.append(Xp_tr)
            y_tr_list.extend([posture] * len(Xp_tr))
            g_tr_list.extend(gp_tr.astype(object).tolist())

        if len(Xp_val) > 0:
            X_val_list.append(Xp_val)
            y_val_list.extend([posture] * len(Xp_val))
            g_val_list.extend(gp_val.astype(object).tolist())

        if len(Xp_te) > 0:
            X_te_list.append(Xp_te)
            y_te_list.extend([posture] * len(Xp_te))
            g_te_list.extend(gp_te.astype(object).tolist())

    def _stack_or_empty(lst: List[np.ndarray]) -> np.ndarray:
        if lst:
            return np.vstack(lst).astype(np.float32)
        return np.zeros((0, feat_dim), dtype=np.float32)

    X_train = _stack_or_empty(X_tr_list)
    y_train = np.array(y_tr_list, dtype=object)
    g_train = np.array(g_tr_list, dtype=object)

    X_val = _stack_or_empty(X_val_list)
    y_val = np.array(y_val_list, dtype=object)
    g_val = np.array(g_val_list, dtype=object)

    X_test = _stack_or_empty(X_te_list)
    y_test = np.array(y_te_list, dtype=object)
    g_test = np.array(g_te_list, dtype=object)

    if len(X_train) <= 0:
        raise RuntimeError("训练集样本数为 0,无法训练.")
    if len(X_test) <= 0:
        raise RuntimeError("测试集样本数为 0,无法评估/训练.")

    if split_mode == "participant_manual":
        split_method = (
            "ParticipantManualSplit("
            f"participants_total={participant_total}, "
            f"participants_used={participant_cnt}, "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}, "
            f"base_posture={base_posture}, seed={seed})"
        )
        split_debug: Dict[str, object] = {
            "split_mode": split_mode,
            "base_posture": base_posture,
            "seed": int(seed),
            "participants_total": int(participant_total),
            "participants_used": int(participant_cnt),
            "participants_all": list(base_ids_all),
            "participants_selected": list(selected_ids),
            "participants_train": sorted(list(train_ids), key=natural_sort_key),
            "participants_val": sorted(list(val_ids), key=natural_sort_key),
            "participants_test": sorted(list(test_ids), key=natural_sort_key),
            "participants_unused": sorted(
                [p for p in base_ids_all if (p not in train_ids and p not in val_ids and p not in test_ids)],
                key=natural_sort_key,
            ),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
        }
    else:
        split_method = (
            "SampleRatioSplitPerPosture("
            "train=0.70,val=0.15,test=0.15,remainder->train, "
            f"participants_total={participant_total}, participants_used={participant_cnt}, "
            f"base_posture={base_posture}, seed={seed})"
        )
        split_debug = {
            "split_mode": split_mode,
            "base_posture": base_posture,
            "seed": int(seed),
            "participants_total": int(participant_total),
            "participants_used": int(participant_cnt),
            "participants_selected": list(selected_ids),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
        }

    return (
        X_train, y_train, g_train,
        X_val, y_val, g_val,
        X_test, y_test, g_test,
        split_method, split_debug,
    )


# ============================================================
# 5) 训练配置:ANN/MLP 超参数
# ============================================================

@dataclass
class ModelConfig:
    hidden_layers: List[int] = field(default_factory=lambda: list(DEFAULT_ANN_HIDDEN_LAYERS))
    activation: str = DEFAULT_ANN_ACTIVATION
    dropout: float = DEFAULT_ANN_DROPOUT

    # FSR augmentation
    enable_fsr_sensor_masking: bool = ENABLE_FSR_SENSOR_MASKING
    fsr_sensor_mask_prob: float = FSR_SENSOR_MASK_PROB

    enable_fsr_domain_randomization: bool = ENABLE_FSR_DOMAIN_RANDOMIZATION
    fsr_dr_scale_range: Tuple[float, float] = FSR_DR_SCALE_RANGE
    fsr_dr_noise_std: float = FSR_DR_NOISE_STD
    fsr_dr_bias_std: float = FSR_DR_BIAS_STD

    # Feature weighting
    enable_feature_weighting: bool = ENABLE_FEATURE_WEIGHTING
    tof_gated_weight: float = TOF_GATED_WEIGHT
    tof_use_weight: float = TOF_USE_WEIGHT

    batch_size: int = DEFAULT_ANN_BATCH_SIZE
    epochs: int = DEFAULT_ANN_EPOCHS
    lr: float = DEFAULT_ANN_LR
    weight_decay: float = DEFAULT_ANN_WEIGHT_DECAY

    lr_scheduler: str = DEFAULT_ANN_LR_SCHEDULER
    lr_min: float = DEFAULT_ANN_LR_MIN
    lr_plateau_factor: float = DEFAULT_ANN_LR_PLATEAU_FACTOR
    lr_plateau_patience: int = DEFAULT_ANN_LR_PLATEAU_PATIENCE
    lr_plateau_threshold: float = DEFAULT_ANN_LR_PLATEAU_THRESHOLD
    lr_plateau_cooldown: int = DEFAULT_ANN_LR_PLATEAU_COOLDOWN

    wd_scheduler: str = DEFAULT_ANN_WD_SCHEDULER
    weight_decay_end: float = DEFAULT_ANN_WEIGHT_DECAY_END

    early_stop_patience: int = DEFAULT_ANN_EARLY_STOP_PATIENCE
    min_delta: float = DEFAULT_ANN_MIN_DELTA
    use_class_weights: bool = DEFAULT_ANN_USE_CLASS_WEIGHTS

    device: str = DEFAULT_ANN_DEVICE
    verbose: int = DEFAULT_ANN_VERBOSE
    random_state: int = DEFAULT_SEED


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn 风格封装,内部用 PyTorch 训练/推理 MLP(ANN).
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = tuple(DEFAULT_ANN_HIDDEN_LAYERS),
        activation: str = DEFAULT_ANN_ACTIVATION,
        dropout: float = DEFAULT_ANN_DROPOUT,

        # FSR augmentation
        enable_fsr_sensor_masking: bool = ENABLE_FSR_SENSOR_MASKING,
        fsr_sensor_mask_prob: float = FSR_SENSOR_MASK_PROB,
        enable_fsr_domain_randomization: bool = ENABLE_FSR_DOMAIN_RANDOMIZATION,
        fsr_dr_scale_range: Tuple[float, float] = FSR_DR_SCALE_RANGE,
        fsr_dr_noise_std: float = FSR_DR_NOISE_STD,
        fsr_dr_bias_std: float = FSR_DR_BIAS_STD,

        fsr_indices: Tuple[int, ...] = (),
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,

        # Feature weighting
        enable_feature_weighting: bool = ENABLE_FEATURE_WEIGHTING,
        feature_weight_indices: Tuple[int, ...] = (),
        feature_weight_values: Tuple[float, ...] = (),

        batch_size: int = DEFAULT_ANN_BATCH_SIZE,
        epochs: int = DEFAULT_ANN_EPOCHS,
        lr: float = DEFAULT_ANN_LR,
        weight_decay: float = DEFAULT_ANN_WEIGHT_DECAY,

        lr_scheduler: str = DEFAULT_ANN_LR_SCHEDULER,
        lr_min: float = DEFAULT_ANN_LR_MIN,
        lr_plateau_factor: float = DEFAULT_ANN_LR_PLATEAU_FACTOR,
        lr_plateau_patience: int = DEFAULT_ANN_LR_PLATEAU_PATIENCE,
        lr_plateau_threshold: float = DEFAULT_ANN_LR_PLATEAU_THRESHOLD,
        lr_plateau_cooldown: int = DEFAULT_ANN_LR_PLATEAU_COOLDOWN,

        wd_scheduler: str = DEFAULT_ANN_WD_SCHEDULER,
        weight_decay_end: float = DEFAULT_ANN_WEIGHT_DECAY_END,

        early_stop_patience: int = DEFAULT_ANN_EARLY_STOP_PATIENCE,
        min_delta: float = DEFAULT_ANN_MIN_DELTA,
        use_class_weights: bool = DEFAULT_ANN_USE_CLASS_WEIGHTS,

        device: str = DEFAULT_ANN_DEVICE,
        verbose: int = DEFAULT_ANN_VERBOSE,
        random_state: int = DEFAULT_SEED,
    ):
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.activation = str(activation)
        self.dropout = float(dropout)

        self.enable_fsr_sensor_masking = bool(enable_fsr_sensor_masking)
        self.fsr_sensor_mask_prob = float(fsr_sensor_mask_prob)

        self.enable_fsr_domain_randomization = bool(enable_fsr_domain_randomization)
        self.fsr_dr_scale_range = tuple(float(v) for v in tuple(fsr_dr_scale_range))
        self.fsr_dr_noise_std = float(fsr_dr_noise_std)
        self.fsr_dr_bias_std = float(fsr_dr_bias_std)

        self.fsr_indices = tuple(int(i) for i in tuple(fsr_indices)) if fsr_indices else ()
        self.scaler_mean = None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float32).reshape(-1)
        self.scaler_scale = None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float32).reshape(-1)

        self.enable_feature_weighting = bool(enable_feature_weighting)
        self.feature_weight_indices = tuple(int(i) for i in (feature_weight_indices or ()))
        self.feature_weight_values = tuple(float(w) for w in (feature_weight_values or ()))
        if len(self.feature_weight_indices) != len(self.feature_weight_values):
            raise ValueError("feature_weight_indices 与 feature_weight_values 长度不一致")

        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.lr_scheduler = str(lr_scheduler)
        self.lr_min = float(lr_min)
        self.lr_plateau_factor = float(lr_plateau_factor)
        self.lr_plateau_patience = int(lr_plateau_patience)
        self.lr_plateau_threshold = float(lr_plateau_threshold)
        self.lr_plateau_cooldown = int(lr_plateau_cooldown)

        self.wd_scheduler = str(wd_scheduler)
        self.weight_decay_end = float(weight_decay_end)

        self.early_stop_patience = int(early_stop_patience)
        self.min_delta = float(min_delta)
        self.use_class_weights = bool(use_class_weights)

        self.device = str(device)
        self.verbose = int(verbose)
        self.random_state = int(random_state)

        self.model_: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.label_to_index_: Optional[Dict[object, int]] = None
        self.history_: Dict[str, List[float]] = {}
        self.best_epoch_: Optional[int] = None
        self.best_val_loss_: Optional[float] = None
        self.epochs_trained_: int = 0
        self.last_lr_: Optional[float] = None
        self.last_weight_decay_: Optional[float] = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        d = str(device).strip().lower()
        if d == "cpu":
            return torch.device("cpu")
        if d == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            print("[WARN] 你选择了 GPU,但当前环境未检测到可用 CUDA/MPS,将回退到 CPU.")
            return torch.device("cpu")
        raise ValueError("ann_device/DEFAULT_ANN_DEVICE 只能是 'CPU' 或 'GPU'.")

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        n = str(name).lower().strip()
        if n == "relu":
            return nn.ReLU()
        if n == "tanh":
            return nn.Tanh()
        if n == "sigmoid":
            return nn.Sigmoid()
        raise ValueError("activation 仅支持 'relu' / 'tanh' / 'sigmoid'.")

    def _create_lr_scheduler(self, optimizer: optim.Optimizer):
        name = str(self.lr_scheduler).lower().strip()
        if name in ("none", "", "false", "0"):
            return None
        if name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.lr_plateau_factor),
                patience=int(self.lr_plateau_patience),
                threshold=float(self.lr_plateau_threshold),
                cooldown=int(self.lr_plateau_cooldown),
                min_lr=float(self.lr_min),
            )
        raise ValueError("lr_scheduler 仅支持 'none' / 'plateau'.")

    def _weight_decay_for_epoch(self, epoch: int, max_epochs: int) -> float:
        mode = str(self.wd_scheduler).lower().strip()
        wd0 = float(self.weight_decay)
        wd1 = float(self.weight_decay_end)
        if (mode in ("none", "", "false", "0")) or (max_epochs <= 1):
            return wd0
        if mode == "linear":
            t = (float(epoch) - 1.0) / float(max_epochs - 1)
            wd = wd0 + t * (wd1 - wd0)
            return max(0.0, float(wd))
        raise ValueError("wd_scheduler 仅支持 'none' / 'linear'.")

    @staticmethod
    def _get_current_lr(optimizer: optim.Optimizer) -> float:
        return float(optimizer.param_groups[0].get("lr", 0.0))

    @staticmethod
    def _get_current_weight_decay(optimizer: optim.Optimizer) -> float:
        return float(optimizer.param_groups[0].get("weight_decay", 0.0))

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for h in self.hidden_layers:
            h = int(h)
            if h <= 0:
                raise ValueError("hidden_layers 中的神经元数量必须为正整数.")
            layers.append(nn.Linear(in_dim, h))
            layers.append(self._make_activation(self.activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=float(self.dropout)))
            in_dim = h
        layers.append(nn.Linear(in_dim, int(output_dim)))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        classes = sorted(np.unique(y).tolist(), key=natural_sort_key)
        self.classes_ = np.array(classes, dtype=object)
        self.label_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.label_to_index_[v] for v in y], dtype=np.int64)

        device = self._resolve_device(self.device)
        torch.manual_seed(self.random_state)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        n = int(len(X))
        if n < 2:
            raise RuntimeError("训练样本数太少,无法训练 ANN.")

        use_val = (X_val is not None) and (y_val is not None) and (len(X_val) > 0)
        X_tr, y_tr = X, y_idx

        if use_val:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val)

            if len(X_val_arr) != len(y_val_arr):
                raise ValueError(f"X_val/y_val 长度不一致: {len(X_val_arr)} vs {len(y_val_arr)}")
            if X_val_arr.ndim != 2 or X_val_arr.shape[1] != X.shape[1]:
                raise ValueError(f"X_val 维度不匹配: {X_val_arr.shape}, 期望 (*,{X.shape[1]})")

            try:
                y_val_idx = np.array([self.label_to_index_[v] for v in y_val_arr], dtype=np.int64)
            except KeyError as e:
                raise ValueError(f"验证集出现训练集中不存在的类别:{e}")

            X_val_final, y_val_final = X_val_arr, y_val_idx
        else:
            X_val_final, y_val_final = None, None

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        train_loader = DataLoader(train_ds, batch_size=max(1, self.batch_size), shuffle=True, drop_last=False)

        if use_val and X_val_final is not None and y_val_final is not None and len(X_val_final) > 0:
            val_ds = TensorDataset(torch.from_numpy(X_val_final), torch.from_numpy(y_val_final))
            val_loader = DataLoader(val_ds, batch_size=max(1, self.batch_size), shuffle=False, drop_last=False)
        else:
            val_loader = None

        model = self._build_model(input_dim=X.shape[1], output_dim=len(self.classes_)).to(device)

        weight_tensor = None
        if self.use_class_weights:
            counts = np.bincount(y_tr, minlength=len(self.classes_)).astype(np.float32)
            counts = np.clip(counts, 1.0, None)
            weights = counts.sum() / (len(self.classes_) * counts)
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        lr_scheduler = self._create_lr_scheduler(optimizer)

        # ---------- augmentation setup ----------
        fsr_cols = list(self.fsr_indices)
        do_mask = bool(self.enable_fsr_sensor_masking) and (float(self.fsr_sensor_mask_prob) > 0.0)

        dr_low, dr_high = (1.0, 1.0)
        try:
            if self.fsr_dr_scale_range is not None:
                dr_low = float(tuple(self.fsr_dr_scale_range)[0])
                dr_high = float(tuple(self.fsr_dr_scale_range)[1])
        except Exception:
            dr_low, dr_high = (1.0, 1.0)

        do_dr = bool(self.enable_fsr_domain_randomization) and (
            (abs(dr_low - 1.0) > 1e-12) or (abs(dr_high - 1.0) > 1e-12)
            or (float(self.fsr_dr_noise_std) > 0.0)
            or (float(self.fsr_dr_bias_std) > 0.0)
        )

        use_fsr_aug = bool(fsr_cols) and (do_mask or do_dr)

        mean_t_cpu: Optional[torch.Tensor] = None
        scale_t_cpu: Optional[torch.Tensor] = None

        if use_fsr_aug:
            if self.scaler_mean is None or self.scaler_scale is None:
                raise RuntimeError("启用了 FSR 训练增强,但未提供 scaler_mean/scaler_scale。")
            if len(self.scaler_mean) != int(X.shape[1]) or len(self.scaler_scale) != int(X.shape[1]):
                raise RuntimeError("scaler_mean/scaler_scale 维度与输入维度不匹配.")
            if not (0.0 <= float(self.fsr_sensor_mask_prob) <= 1.0):
                raise ValueError("FSR_SENSOR_MASK_PROB 必须在 [0,1].")
            if dr_low <= 0.0 or dr_high <= 0.0 or dr_low > dr_high:
                raise ValueError("FSR_DR_SCALE_RANGE 必须满足 0 < low <= high.")

            for j in fsr_cols:
                if int(j) < 0 or int(j) >= int(X.shape[1]):
                    raise RuntimeError(f"fsr_indices 包含越界索引: {j}")

            mean_t_cpu = torch.from_numpy(np.asarray(self.scaler_mean, dtype=np.float32)).view(1, -1)
            scale_t_cpu = torch.from_numpy(np.asarray(self.scaler_scale, dtype=np.float32)).view(1, -1)

        def _apply_fsr_aug_on_standardized_cpu(xb_cpu: torch.Tensor) -> torch.Tensor:
            if (not use_fsr_aug) or (mean_t_cpu is None) or (scale_t_cpu is None):
                return xb_cpu
            x_raw = xb_cpu * scale_t_cpu + mean_t_cpu
            fsr = x_raw[:, fsr_cols].clone()

            if do_dr:
                if abs(dr_low - 1.0) > 1e-12 or abs(dr_high - 1.0) > 1e-12:
                    gain = torch.empty_like(fsr).uniform_(dr_low, dr_high)
                    fsr = fsr * gain
                if float(self.fsr_dr_noise_std) > 0.0:
                    fsr = fsr + torch.randn_like(fsr) * float(self.fsr_dr_noise_std)
                if float(self.fsr_dr_bias_std) > 0.0:
                    b = torch.randn((fsr.shape[0], 1), dtype=fsr.dtype) * float(self.fsr_dr_bias_std)
                    fsr = fsr + b

            if do_mask:
                mask = torch.rand_like(fsr) < float(self.fsr_sensor_mask_prob)
                fsr = fsr.masked_fill(mask, 0.0)

            x_raw[:, fsr_cols] = fsr
            xb_out = (x_raw - mean_t_cpu) / scale_t_cpu
            return xb_out

        def _apply_feature_weighting_cpu(xb_cpu: torch.Tensor) -> torch.Tensor:
            if (not self.enable_feature_weighting) or (not self.feature_weight_indices):
                return xb_cpu
            xw = xb_cpu
            for idx, w in zip(self.feature_weight_indices, self.feature_weight_values):
                ii = int(idx)
                if 0 <= ii < xw.shape[1]:
                    xw[:, ii] = xw[:, ii] * float(w)
            return xw

        # ---------- train loop ----------
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "weight_decay": []}
        best_state = None
        best_val = float("inf")
        best_epoch = -1
        patience_left = int(self.early_stop_patience)

        max_epochs = int(max(1, self.epochs))
        min_delta = float(self.min_delta)

        for epoch in range(1, max_epochs + 1):
            wd_now = self._weight_decay_for_epoch(epoch=epoch, max_epochs=max_epochs)
            for pg in optimizer.param_groups:
                pg["weight_decay"] = float(wd_now)

            model.train()
            total_loss = 0.0
            total_n = 0

            for xb, yb in train_loader:
                xb = _apply_fsr_aug_on_standardized_cpu(xb)
                xb = _apply_feature_weighting_cpu(xb)

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = int(xb.shape[0])
                total_loss += float(loss.item()) * bs
                total_n += bs

            train_loss = total_loss / max(1, total_n)
            self.history_["train_loss"].append(float(train_loss))

            val_loss = None
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                v_n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = _apply_feature_weighting_cpu(xb)
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        bs = int(xb.shape[0])
                        v_loss += float(loss.item()) * bs
                        v_n += bs
                val_loss = v_loss / max(1, v_n)
                self.history_["val_loss"].append(float(val_loss))

                improved = (val_loss < (best_val - min_delta))
                if improved:
                    best_val = float(val_loss)
                    best_epoch = int(epoch)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = int(self.early_stop_patience)
                else:
                    if self.early_stop_patience > 0:
                        patience_left -= 1

                if lr_scheduler is not None:
                    lr_scheduler.step(float(val_loss))

                cur_lr = self._get_current_lr(optimizer)
                cur_wd = self._get_current_weight_decay(optimizer)
                self.history_["lr"].append(float(cur_lr))
                self.history_["weight_decay"].append(float(cur_wd))
                self.last_lr_ = float(cur_lr)
                self.last_weight_decay_ = float(cur_wd)

                if self.verbose:
                    print(
                        f"[ANN] epoch {epoch:03d}/{max_epochs}  "
                        f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                        f"lr={cur_lr:.3e}  wd={cur_wd:.3e}"
                    )

                if self.early_stop_patience > 0 and patience_left <= 0:
                    if self.verbose:
                        print(f"[ANN] Early stopping at epoch={epoch} (best_epoch={best_epoch}, best_val={best_val:.6f})")
                    self.epochs_trained_ = int(epoch)
                    break
            else:
                if lr_scheduler is not None:
                    lr_scheduler.step(float(train_loss))

                cur_lr = self._get_current_lr(optimizer)
                cur_wd = self._get_current_weight_decay(optimizer)
                self.history_["lr"].append(float(cur_lr))
                self.history_["weight_decay"].append(float(cur_wd))
                self.last_lr_ = float(cur_lr)
                self.last_weight_decay_ = float(cur_wd)

                if self.verbose:
                    print(
                        f"[ANN] epoch {epoch:03d}/{max_epochs}  "
                        f"train_loss={train_loss:.6f}  lr={cur_lr:.3e}  wd={cur_wd:.3e}"
                    )
                self.epochs_trained_ = int(epoch)

        if best_state is not None:
            model.load_state_dict(best_state)
            self.best_epoch_ = int(best_epoch)
            self.best_val_loss_ = float(best_val)
        else:
            self.best_epoch_ = None
            self.best_val_loss_ = None

        self.model_ = model
        if self.epochs_trained_ <= 0:
            self.epochs_trained_ = int(max_epochs)

        return self

    def _apply_feature_weighting_numpy(self, X: np.ndarray) -> np.ndarray:
        if (not self.enable_feature_weighting) or (not self.feature_weight_indices):
            return X
        Xw = np.asarray(X, dtype=np.float32).copy()
        for idx, w in zip(self.feature_weight_indices, self.feature_weight_values):
            ii = int(idx)
            if 0 <= ii < Xw.shape[1]:
                Xw[:, ii] *= float(w)
        return Xw

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("模型尚未训练,请先调用 fit().")
        X = np.asarray(X, dtype=np.float32)
        X = self._apply_feature_weighting_numpy(X)

        device = self._resolve_device(self.device)
        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(device)
            logits = self.model_(xb)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        return self.classes_[pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型尚未训练,请先调用 fit().")
        X = np.asarray(X, dtype=np.float32)
        X = self._apply_feature_weighting_numpy(X)

        device = self._resolve_device(self.device)
        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(device)
            logits = self.model_(xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        return probs


def train_and_evaluate(
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    g_test: np.ndarray,
    model_cfg: ModelConfig,
    seed: int,
    split_method: str,
    split_debug: Dict[str, object],
) -> Tuple[Pipeline, Dict[str, object], Dict[str, object]]:
    if len(X_train) <= 0:
        raise RuntimeError("训练集为空,无法训练.")
    if len(X_test) <= 0:
        raise RuntimeError("测试集为空,无法评估.")

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    use_external_val = (X_val is not None) and (len(X_val) > 0) and (y_val is not None) and (len(y_val) > 0)
    if use_external_val:
        X_val_s = scaler.transform(X_val).astype(np.float32)
        y_val_use = y_val
    else:
        X_val_s = None
        y_val_use = None

    if len(feature_names) != int(X_train.shape[1]):
        raise ValueError(f"feature_names 长度与输入维度不一致: len={len(feature_names)} vs D={int(X_train.shape[1])}")

    fsr_indices = tuple(i for i, n in enumerate(feature_names) if n in set(FSR_FEATURE_NAMES))

    # Feature weighting indices (StandardScaler 之后乘权重)
    fw_indices: List[int] = []
    fw_values: List[float] = []
    if bool(model_cfg.enable_feature_weighting):
        if "tof_gated" in feature_names:
            fw_indices.append(int(feature_names.index("tof_gated")))
            fw_values.append(float(model_cfg.tof_gated_weight))
        if "tof_use" in feature_names and abs(float(model_cfg.tof_use_weight) - 1.0) > 1e-12:
            fw_indices.append(int(feature_names.index("tof_use")))
            fw_values.append(float(model_cfg.tof_use_weight))

    clf = TorchMLPClassifier(
        hidden_layers=tuple(model_cfg.hidden_layers),
        activation=model_cfg.activation,
        dropout=model_cfg.dropout,

        enable_fsr_sensor_masking=bool(model_cfg.enable_fsr_sensor_masking),
        fsr_sensor_mask_prob=float(model_cfg.fsr_sensor_mask_prob),
        enable_fsr_domain_randomization=bool(model_cfg.enable_fsr_domain_randomization),
        fsr_dr_scale_range=tuple(model_cfg.fsr_dr_scale_range),
        fsr_dr_noise_std=float(model_cfg.fsr_dr_noise_std),
        fsr_dr_bias_std=float(model_cfg.fsr_dr_bias_std),

        enable_feature_weighting=bool(model_cfg.enable_feature_weighting),
        feature_weight_indices=tuple(fw_indices),
        feature_weight_values=tuple(fw_values),

        fsr_indices=fsr_indices,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),

        batch_size=model_cfg.batch_size,
        epochs=model_cfg.epochs,
        lr=model_cfg.lr,
        weight_decay=model_cfg.weight_decay,

        lr_scheduler=model_cfg.lr_scheduler,
        lr_min=model_cfg.lr_min,
        lr_plateau_factor=model_cfg.lr_plateau_factor,
        lr_plateau_patience=model_cfg.lr_plateau_patience,
        lr_plateau_threshold=model_cfg.lr_plateau_threshold,
        lr_plateau_cooldown=model_cfg.lr_plateau_cooldown,

        wd_scheduler=model_cfg.wd_scheduler,
        weight_decay_end=model_cfg.weight_decay_end,

        early_stop_patience=model_cfg.early_stop_patience,
        min_delta=model_cfg.min_delta,
        use_class_weights=model_cfg.use_class_weights,

        device=model_cfg.device,
        verbose=model_cfg.verbose,
        random_state=int(seed),
    )

    clf.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val_use)

    pipeline = Pipeline(steps=[
        ("scaler", scaler),
        ("clf", clf),
    ])

    y_pred = clf.predict(X_test_s)
    acc = float(accuracy_score(y_test, y_pred))
    report_text = classification_report(y_test, y_pred, digits=4, output_dict=False)

    classes_order = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes_order)
    test_counts = [int(np.sum(y_test == lbl)) for lbl in classes_order]

    y_all = np.concatenate([y_train, y_val, y_test]) if len(y_val) > 0 else np.concatenate([y_train, y_test])
    g_all = np.concatenate([g_train, g_val, g_test]) if len(g_val) > 0 else np.concatenate([g_train, g_test])
    classes = np.unique(y_all)
    group_cnt_per_class = {c: int(len(np.unique(g_all[y_all == c]))) for c in classes}

    metrics = {
        "model_type": "ann",
        "accuracy": acc,
        "classification_report_text": report_text,
        "labels": classes_order,
        "confusion_matrix": cm,
        "test_counts": test_counts,
        "split_method": split_method,
        "split_debug": split_debug,
        "group_cnt_per_class": group_cnt_per_class,
        "ann": {
            "hidden_layers": list(model_cfg.hidden_layers),
            "activation": model_cfg.activation,
            "dropout": model_cfg.dropout,

            "fsr_sensor_masking": bool(model_cfg.enable_fsr_sensor_masking),
            "fsr_sensor_mask_prob": float(model_cfg.fsr_sensor_mask_prob),
            "fsr_domain_randomization": bool(model_cfg.enable_fsr_domain_randomization),
            "fsr_dr_scale_range": [float(model_cfg.fsr_dr_scale_range[0]), float(model_cfg.fsr_dr_scale_range[1])],
            "fsr_dr_noise_std": float(model_cfg.fsr_dr_noise_std),
            "fsr_dr_bias_std": float(model_cfg.fsr_dr_bias_std),
            "fsr_indices": [int(i) for i in fsr_indices],

            "feature_weighting": bool(model_cfg.enable_feature_weighting),
            "feature_weight_indices": [int(i) for i in fw_indices],
            "feature_weight_values": [float(w) for w in fw_values],

            "batch_size": model_cfg.batch_size,
            "epochs_config": model_cfg.epochs,
            "epochs_trained": int(clf.epochs_trained_),
            "best_epoch": None if clf.best_epoch_ is None else int(clf.best_epoch_),
            "best_val_loss": None if clf.best_val_loss_ is None else float(clf.best_val_loss_),

            "use_class_weights": bool(model_cfg.use_class_weights),
            "device": str(model_cfg.device),

            "lr_init": float(model_cfg.lr),
            "weight_decay_init": float(model_cfg.weight_decay),
            "lr_scheduler": str(model_cfg.lr_scheduler),
            "lr_min": float(model_cfg.lr_min),
            "wd_scheduler": str(model_cfg.wd_scheduler),
            "weight_decay_end": float(model_cfg.weight_decay_end),
            "lr_final": None if clf.last_lr_ is None else float(clf.last_lr_),
            "weight_decay_final": None if clf.last_weight_decay_ is None else float(clf.last_weight_decay_),
        },
    }

    extras = {
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "use_external_val": bool(use_external_val),
        "unique_groups_train": int(len(np.unique(g_train))) if len(g_train) > 0 else 0,
        "unique_groups_val": int(len(np.unique(g_val))) if len(g_val) > 0 else 0,
        "unique_groups_test": int(len(np.unique(g_test))) if len(g_test) > 0 else 0,
    }
    return pipeline, metrics, extras


# ============================================================
# 5.5) 可选:按 participant 的交叉验证
# ============================================================

def run_group_cross_validation(
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_cfg: ModelConfig,
    seed: int,
    method: str,
    n_splits: int,
    out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    if X is None or len(X) == 0:
        print("[CV][WARN] 数据为空,跳过交叉验证.")
        return {"enabled": True, "skipped": True, "reason": "empty_data"}

    if groups is None or len(groups) == 0:
        print("[CV][WARN] groups 为空,跳过交叉验证.")
        return {"enabled": True, "skipped": True, "reason": "empty_groups"}

    uniq_groups = np.unique(groups)
    n_groups = int(len(uniq_groups))
    if n_groups < 2:
        print(f"[CV][WARN] unique participant 数量 < 2 (n_groups={n_groups}),跳过.")
        return {"enabled": True, "skipped": True, "reason": "insufficient_groups", "n_groups": n_groups}

    method_norm = str(method).strip().upper()
    if method_norm in ("GROUPKFOLD", "GK", "GROUP_KFOLD"):
        k = int(max(2, n_splits))
        if k > n_groups:
            print(f"[CV][WARN] n_splits={k} > unique_groups={n_groups},将自动设置为 {n_groups}.")
            k = n_groups
        splitter = GroupKFold(n_splits=k)
        method_desc = f"GroupKFold(n_splits={k})"
    elif method_norm in ("LEAVEONEGROUPOUT", "LOGO", "LOSO", "LEAVE_ONE_GROUP_OUT"):
        splitter = LeaveOneGroupOut()
        method_desc = "LeaveOneGroupOut"
    else:
        raise ValueError("CROSS_VALIDATION_METHOD 只能是 'GroupKFold' 或 'LeaveOneGroupOut'.")

    global_labels = sorted([str(x) for x in np.unique(y)])
    label_to_idx = {lbl: i for i, lbl in enumerate(global_labels)}
    cm_sum = np.zeros((len(global_labels), len(global_labels)), dtype=np.int64)
    test_counts_sum = np.zeros((len(global_labels),), dtype=np.int64)

    acc_list: List[float] = []
    test_n_list: List[int] = []
    fold_items: List[Dict[str, object]] = []

    empty_X = np.zeros((0, int(X.shape[1])), dtype=np.float32)
    empty_y = np.array([], dtype=y.dtype)
    empty_g = np.array([], dtype=groups.dtype)

    n_folds = 0
    print("\n================ Cross Validation ================")
    print(f"CV method: {method_desc}")
    print(f"Unique participants(groups): {n_groups}")
    print("--------------------------------------------------")

    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        n_folds += 1
        X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
        X_te, y_te, g_te = X[te_idx], y[te_idx], groups[te_idx]

        fold_seed = int(seed) + int(fold_idx)
        split_method = f"cv::{method_desc}::fold_{fold_idx}"
        split_debug = {
            "split_mode": "cross_validation",
            "cv_method": method_desc,
            "fold": int(fold_idx),
            "train_groups_count": int(len(np.unique(g_tr))),
            "test_groups_count": int(len(np.unique(g_te))),
            "train_samples": int(len(y_tr)),
            "test_samples": int(len(y_te)),
        }

        _, metrics, _ = train_and_evaluate(
            feature_names=feature_names,
            X_train=X_tr, y_train=y_tr, g_train=g_tr,
            X_val=empty_X, y_val=empty_y, g_val=empty_g,
            X_test=X_te, y_test=y_te, g_test=g_te,
            model_cfg=model_cfg,
            seed=fold_seed,
            split_method=split_method,
            split_debug=split_debug,
        )

        acc = float(metrics.get("accuracy", 0.0))
        acc_list.append(acc)
        test_n_list.append(int(len(y_te)))

        fold_labels = [str(x) for x in metrics.get("labels", [])]
        fold_cm = metrics.get("confusion_matrix", None)
        if fold_cm is not None and len(fold_labels) == fold_cm.shape[0]:
            for i, li in enumerate(fold_labels):
                for j, lj in enumerate(fold_labels):
                    if li in label_to_idx and lj in label_to_idx:
                        cm_sum[label_to_idx[li], label_to_idx[lj]] += int(fold_cm[i, j])

        for lbl in np.unique(y_te):
            s_lbl = str(lbl)
            if s_lbl in label_to_idx:
                test_counts_sum[label_to_idx[s_lbl]] += int(np.sum(y_te == lbl))

        fold_items.append({
            "fold": int(fold_idx),
            "accuracy": float(acc),
            "test_samples": int(len(y_te)),
            "test_groups": sorted([str(x) for x in np.unique(g_te)]),
        })

        print(f"[CV] fold {fold_idx:02d}: acc={acc:.4f}, test_samples={len(y_te)}, test_groups={sorted([str(x) for x in np.unique(g_te)])}")

    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    std_acc = float(np.std(acc_list, ddof=1)) if len(acc_list) > 1 else 0.0
    weighted_acc = float(np.sum(np.asarray(acc_list) * np.asarray(test_n_list)) / max(1, int(np.sum(test_n_list)))) if test_n_list else 0.0

    print("--------------------------------------------------")
    print(f"CV folds: {n_folds}")
    print(f"CV mean accuracy: {mean_acc:.4f}")
    print(f"CV std  accuracy: {std_acc:.4f}")
    print(f"CV weighted accuracy: {weighted_acc:.4f}")
    print("==================================================\n")

    summary: Dict[str, object] = {
        "enabled": True,
        "method": method_desc,
        "n_folds": int(n_folds),
        "n_groups": int(n_groups),
        "labels": global_labels,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "weighted_accuracy": weighted_acc,
        "folds": fold_items,
        "confusion_matrix": cm_sum.astype(int).tolist(),
        "test_counts": test_counts_sum.astype(int).tolist(),
    }

    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv_json_path = out_dir / OUT_CV_SUMMARY_NAME
            cv_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            cv_cm_path = out_dir / OUT_CV_CM_NAME
            save_confusion_matrix_standard(
                cm=cm_sum,
                labels=global_labels,
                test_counts=[int(x) for x in test_counts_sum.tolist()],
                out_path=cv_cm_path,
            )
            print(f"[CV] summary saved to: {cv_json_path}")
            print(f"[CV] confusion matrix saved to: {cv_cm_path}")
        except Exception as e:
            print(f"[CV][WARN] 保存交叉验证结果失败(已忽略): {e}")

    return summary


# ============================================================
# 6) 导出:C 头文件
# ============================================================

def _fmt_f(x: float) -> str:
    return f"{float(x):.9g}f"


def export_ann_to_c_header(
    pipeline: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    out_path: Path,
):
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: TorchMLPClassifier = pipeline.named_steps["clf"]

    if clf.model_ is None or clf.classes_ is None:
        raise RuntimeError("导出失败:ANN 模型尚未训练完成(clf.model_ is None).")

    if list(clf.classes_) != class_names:
        class_names = list(clf.classes_)

    n_feat = int(len(scaler.mean_))
    n_cls = int(len(class_names))

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    linear_layers: List[nn.Linear] = []
    for m in clf.model_:
        if isinstance(m, nn.Linear):
            linear_layers.append(m)

    if not linear_layers:
        raise RuntimeError("导出失败:在 ANN 模型中未找到任何 nn.Linear 层.")

    hidden = list(clf.hidden_layers)
    layer_sizes = [n_feat] + hidden + [n_cls]

    if len(linear_layers) != (len(layer_sizes) - 1):
        raise RuntimeError(f"导出失败:Linear 层数量({len(linear_layers)})与 layer_sizes({layer_sizes})不匹配.")

    act = str(clf.activation).lower().strip()
    if act not in ("relu", "tanh", "sigmoid"):
        raise ValueError("目前导出仅支持 activation='relu' / 'tanh' / 'sigmoid'.")

    w_offsets = [0]
    b_offsets = [0]
    w_flat: List[np.ndarray] = []
    b_flat: List[np.ndarray] = []

    for li, lin in enumerate(linear_layers):
        W = lin.weight.detach().cpu().numpy().astype(np.float32)
        B = lin.bias.detach().cpu().numpy().astype(np.float32)
        out_dim, in_dim = W.shape
        if in_dim != layer_sizes[li] or out_dim != layer_sizes[li + 1]:
            raise RuntimeError(f"导出失败:第 {li} 层形状不匹配: W.shape={W.shape}, layer_sizes={layer_sizes}")
        w_flat.append(W.reshape(-1))
        b_flat.append(B.reshape(-1))
        w_offsets.append(w_offsets[-1] + int(W.size))
        b_offsets.append(b_offsets[-1] + int(B.size))

    W_ALL = np.concatenate(w_flat).astype(np.float32)
    B_ALL = np.concatenate(b_flat).astype(np.float32)

    total_w = int(W_ALL.size)
    total_b = int(B_ALL.size)
    max_units = int(max(layer_sizes))
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Feature weighting: StandardScaler 后乘权重 (MCU 端也要一致)
    fw_indices: List[int] = []
    fw_values: List[float] = []
    if ENABLE_FEATURE_WEIGHTING:
        if "tof_gated" in feature_names:
            fw_indices.append(int(feature_names.index("tof_gated")))
            fw_values.append(float(TOF_GATED_WEIGHT))
        if "tof_use" in feature_names and abs(float(TOF_USE_WEIGHT) - 1.0) > 1e-12:
            fw_indices.append(int(feature_names.index("tof_use")))
            fw_values.append(float(TOF_USE_WEIGHT))

    lines: List[str] = []
    lines.append(f"// Auto-generated by AML_ANN.py (ANN mode) on {now}")
    lines.append(f"// Input: {n_feat} features (no feature engineering on MCU)")
    lines.append("// Model: MLP / ANN (PyTorch)")
    lines.append(f"// Hidden layers: {hidden}")
    lines.append(f"// Activation: {act}")
    lines.append("// Feature order expected by aml_predict_class():")
    for i, name in enumerate(feature_names):
        lines.append(f"//   x[{i:2d}] = {name}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("extern \"C\" {")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#define AML_FEATURES {n_feat}")
    lines.append(f"#define AML_NUM_CLASSES {n_cls}")
    lines.append(f"#define AML_ANN_NUM_LAYERS {len(layer_sizes) - 1}")
    lines.append(f"#define AML_ANN_MAX_UNITS {max_units}")
    lines.append(f"#define AML_ANN_TOTAL_W {total_w}")
    lines.append(f"#define AML_ANN_TOTAL_B {total_b}")
    lines.append(f"#define AML_FEATURE_WEIGHT_COUNT {len(fw_indices)}")
    lines.append("")

    lines.append("static const char* AML_CLASS_NAMES[AML_NUM_CLASSES] = {")
    for c in class_names:
        lines.append(f"  \"{c}\",")
    lines.append("};")

    lines.append("")
    lines.append(
        f"static const float AML_SCALER_MEAN[AML_FEATURES] = {{ " + ", ".join(_fmt_f(v) for v in mean) + " };"
    )
    lines.append(
        f"static const float AML_SCALER_SCALE[AML_FEATURES] = {{ " + ", ".join(_fmt_f(v) for v in scale) + " };"
    )

    lines.append("")
    lines.append(
        "static const int16_t AML_ANN_LAYER_SIZES[AML_ANN_NUM_LAYERS + 1] = { "
        + ", ".join(str(int(s)) for s in layer_sizes)
        + " };"
    )

    lines.append("")
    lines.append(
        "static const int32_t AML_ANN_W_OFFSET[AML_ANN_NUM_LAYERS + 1] = { "
        + ", ".join(str(int(o)) for o in w_offsets)
        + " };"
    )
    lines.append(
        "static const int32_t AML_ANN_B_OFFSET[AML_ANN_NUM_LAYERS + 1] = { "
        + ", ".join(str(int(o)) for o in b_offsets)
        + " };"
    )

    lines.append("")
    lines.append("static const int16_t AML_FEATURE_WEIGHT_INDICES[AML_FEATURE_WEIGHT_COUNT] = { " +
                 (", ".join(str(int(i)) for i in fw_indices) if fw_indices else "") + " };")
    lines.append("static const float AML_FEATURE_WEIGHT_VALUES[AML_FEATURE_WEIGHT_COUNT] = { " +
                 (", ".join(_fmt_f(v) for v in fw_values) if fw_values else "") + " };")

    lines.append("")
    lines.append("static const float AML_ANN_W[AML_ANN_TOTAL_W] = {")
    for i in range(0, total_w, 8):
        chunk = ", ".join(_fmt_f(v) for v in W_ALL[i:i + 8])
        lines.append(f"  {chunk},")
    lines.append("};")

    lines.append("")
    lines.append("static const float AML_ANN_B[AML_ANN_TOTAL_B] = {")
    for i in range(0, total_b, 8):
        chunk = ", ".join(_fmt_f(v) for v in B_ALL[i:i + 8])
        lines.append(f"  {chunk},")
    lines.append("};")

    lines.append("\n\n// =============================")
    lines.append("// 推理:标准化 + (可选)特征加权 + MLP 前向")
    lines.append("// =============================")

    lines.append("\nstatic inline float aml_ann_activate(float v) {")
    if act == "relu":
        lines.append("  return (v > 0.0f) ? v : 0.0f;")
    elif act == "tanh":
        lines.append("  return tanhf(v);")
    elif act == "sigmoid":
        lines.append("  if (v >= 0.0f) {")
        lines.append("    float z = expf(-v);")
        lines.append("    return 1.0f / (1.0f + z);")
        lines.append("  } else {")
        lines.append("    float z = expf(v);")
        lines.append("    return z / (1.0f + z);")
        lines.append("  }")
    else:
        lines.append("  return v;")
    lines.append("}")

    lines.append("\nstatic inline int aml_predict_class(const float x_in[AML_FEATURES]) {")
    lines.append("  float buf0[AML_ANN_MAX_UNITS];")
    lines.append("  float buf1[AML_ANN_MAX_UNITS];")
    lines.append("")
    lines.append("  // Standardize -> buf0")
    lines.append("  for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("    buf0[j] = (x_in[j] - AML_SCALER_MEAN[j]) / AML_SCALER_SCALE[j];")
    lines.append("  }")
    lines.append("")
    lines.append("  // Feature weighting (after StandardScaler)")
    lines.append("  for (int k = 0; k < AML_FEATURE_WEIGHT_COUNT; ++k) {")
    lines.append("    int idx = (int)AML_FEATURE_WEIGHT_INDICES[k];")
    lines.append("    if (idx >= 0 && idx < AML_FEATURES) {")
    lines.append("      buf0[idx] *= AML_FEATURE_WEIGHT_VALUES[k];")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  float* in = buf0;")
    lines.append("  float* out = buf1;")
    lines.append("")
    lines.append("  for (int l = 0; l < AML_ANN_NUM_LAYERS; ++l) {")
    lines.append("    const int in_size  = (int)AML_ANN_LAYER_SIZES[l];")
    lines.append("    const int out_size = (int)AML_ANN_LAYER_SIZES[l + 1];")
    lines.append("    const int w_off = (int)AML_ANN_W_OFFSET[l];")
    lines.append("    const int b_off = (int)AML_ANN_B_OFFSET[l];")
    lines.append("")
    lines.append("    for (int i = 0; i < out_size; ++i) {")
    lines.append("      float s = AML_ANN_B[b_off + i];")
    lines.append("      const int w_row = w_off + i * in_size;")
    lines.append("      for (int j = 0; j < in_size; ++j) {")
    lines.append("        s += AML_ANN_W[w_row + j] * in[j];")
    lines.append("      }")
    lines.append("      if (l < (AML_ANN_NUM_LAYERS - 1)) s = aml_ann_activate(s);")
    lines.append("      out[i] = s;")
    lines.append("    }")
    lines.append("")
    lines.append("    float* tmp = in; in = out; out = tmp;")
    lines.append("  }")
    lines.append("")
    lines.append("  int best_k = 0;")
    lines.append("  float best_v = in[0];")
    lines.append("  for (int k = 1; k < AML_NUM_CLASSES; ++k) {")
    lines.append("    float v = in[k];")
    lines.append("    if (v > best_v) { best_v = v; best_k = k; }")
    lines.append("  }")
    lines.append("  return best_k;")
    lines.append("}")

    lines.append("\nstatic inline const char* aml_predict_label(const float x_in[AML_FEATURES]) {")
    lines.append("  return AML_CLASS_NAMES[ aml_predict_class(x_in) ];")
    lines.append("}")

    lines.append("\n#ifdef __cplusplus")
    lines.append("} // extern \"C\"")
    lines.append("#endif")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_model_to_c_header(
    pipeline: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    out_path: Path,
):
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, TorchMLPClassifier):
        export_ann_to_c_header(pipeline, feature_names, class_names, out_path)
    else:
        raise TypeError(f"不支持导出的分类器类型:{type(clf)}")


def save_confusion_matrix_standard(
    cm: np.ndarray,
    labels: List[str],
    test_counts: Optional[List[int]],
    out_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    if test_counts is not None:
        pairs = [f"{lbl}={cnt}" for lbl, cnt in zip(labels, test_counts)]
        lines = []
        step = 4
        for i in range(0, len(pairs), step):
            lines.append(", ".join(pairs[i:i + step]))
        counts_str = "\n".join(lines)

        ax.text(
            0.5, -0.2,
            f"Test set counts:\n{counts_str}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8) main
# ============================================================

def main():
    for forbidden in FORBIDDEN_CLI_ARGS:
        if forbidden in sys.argv:
            raise RuntimeError(
                f"已禁用命令行参数 {forbidden}.\n"
                f"请在代码中修改 DATA_ROOT_PATH / OUTPUT_ROOT_PATH."
            )

    parser = argparse.ArgumentParser(
        description="训练坐姿 ANN(MLP) 并导出为 ESP32/Arduino 可用的 C 头文件(tof_gated/tof_use 由采集端提供)"
    )

    parser.add_argument("--postures", type=str, default="")
    parser.add_argument("--max_files_per_posture", type=int, default=DEFAULT_MAX_FILES_PER_POSTURE)
    parser.add_argument("--rows_per_posture", type=int, default=DEFAULT_ROWS_PER_POSTURE)
    parser.add_argument("--rows_per_file", type=int, default=DEFAULT_ROWS_PER_FILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no_confirm", action="store_true", default=DEFAULT_NO_CONFIRM)

    parser.add_argument(
        "--use_tof_features",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="兼容旧版本: 1=强制加入 TOF,0=强制移除 TOF,-1=不覆盖(默认,按 TRAIN_FEATURE_KEYS).",
    )

    parser.add_argument("--ann_hidden_layers", type=str, default=",".join(str(x) for x in DEFAULT_ANN_HIDDEN_LAYERS))
    parser.add_argument("--ann_activation", type=str, default=DEFAULT_ANN_ACTIVATION, choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--ann_dropout", type=float, default=DEFAULT_ANN_DROPOUT)
    parser.add_argument("--ann_batch_size", type=int, default=DEFAULT_ANN_BATCH_SIZE)
    parser.add_argument("--ann_epochs", type=int, default=DEFAULT_ANN_EPOCHS)
    parser.add_argument("--ann_lr", type=float, default=DEFAULT_ANN_LR)
    parser.add_argument("--ann_weight_decay", type=float, default=DEFAULT_ANN_WEIGHT_DECAY)

    parser.add_argument("--ann_lr_scheduler", type=str, default=DEFAULT_ANN_LR_SCHEDULER, choices=["none", "plateau"])
    parser.add_argument("--ann_lr_min", type=float, default=DEFAULT_ANN_LR_MIN)
    parser.add_argument("--ann_lr_plateau_factor", type=float, default=DEFAULT_ANN_LR_PLATEAU_FACTOR)
    parser.add_argument("--ann_lr_plateau_patience", type=int, default=DEFAULT_ANN_LR_PLATEAU_PATIENCE)
    parser.add_argument("--ann_lr_plateau_threshold", type=float, default=DEFAULT_ANN_LR_PLATEAU_THRESHOLD)
    parser.add_argument("--ann_lr_plateau_cooldown", type=int, default=DEFAULT_ANN_LR_PLATEAU_COOLDOWN)

    parser.add_argument("--ann_wd_scheduler", type=str, default=DEFAULT_ANN_WD_SCHEDULER, choices=["none", "linear"])
    parser.add_argument("--ann_weight_decay_end", type=float, default=DEFAULT_ANN_WEIGHT_DECAY_END)
    parser.add_argument("--ann_early_stop_patience", type=int, default=DEFAULT_ANN_EARLY_STOP_PATIENCE)
    parser.add_argument("--ann_min_delta", type=float, default=DEFAULT_ANN_MIN_DELTA)
    parser.add_argument("--ann_use_class_weights", type=int, choices=[0, 1], default=1 if DEFAULT_ANN_USE_CLASS_WEIGHTS else 0)
    parser.add_argument("--ann_device", type=lambda s: str(s).strip().upper(), choices=["CPU", "GPU"], default=str(DEFAULT_ANN_DEVICE).strip().upper())
    parser.add_argument("--ann_verbose", type=int, default=DEFAULT_ANN_VERBOSE, choices=[0, 1])

    args = parser.parse_args()

    data_root = Path(DATA_ROOT_PATH).expanduser().resolve()
    out_dir = Path(OUTPUT_ROOT_PATH).expanduser().resolve()
    if ENABLE_MODEL_EXPORT:
        out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root 不存在:{data_root}")

    detected_postures = auto_detect_postures_folder_only(data_root)
    if not detected_postures:
        raise RuntimeError(f"在 {data_root} 下没有发现任何 posture 子文件夹(或子文件夹内没有 *.csv).")

    if args.postures.strip():
        selected_postures = [p.strip() for p in args.postures.split(",") if p.strip()]
    else:
        selected_postures = detected_postures

    unknown = [p for p in selected_postures if p not in detected_postures]
    if unknown:
        raise RuntimeError("你指定了不存在的 posture 文件夹:" + ", ".join(unknown))

    max_files = args.max_files_per_posture if args.max_files_per_posture > 0 else None
    rows_per_posture = args.rows_per_posture if args.rows_per_posture > 0 else None
    rows_per_file = args.rows_per_file if args.rows_per_file > 0 else None

    feature_keys = list(TRAIN_FEATURE_KEYS)
    if int(args.use_tof_features) == 1:
        if not any(str(k).strip().upper() == "TOF" for k in feature_keys):
            feature_keys.append("TOF")
    elif int(args.use_tof_features) == 0:
        feature_keys = [k for k in feature_keys if str(k).strip().upper() != "TOF"]

    feat_cfg = FeatureBuildConfig(feature_keys=feature_keys)
    model_feature_names = get_model_feature_names(feat_cfg.feature_keys)

    info = scan_dataset_structure(
        data_root=data_root,
        detected_postures=detected_postures,
        selected_postures=selected_postures,
        max_files_per_posture=max_files,
    )

    print("\n================ 数据目录扫描结果(训练前确认) ================")
    print(f"主数据目录 data_root: {data_root}")
    print(f"输出目录 out_dir:      {out_dir}")
    print(f"检测到 posture 文件夹数量: {len(detected_postures)}")
    print("检测到的 posture:", detected_postures)
    print("本次用于训练的 posture:", selected_postures)

    print("\n每个 posture 的 participant 文件数量(used/total):")
    for posture in detected_postures:
        total_files = info[posture]["total_files"]
        used_files = info[posture]["used_files"]
        mark = "  [TRAIN]" if posture in set(selected_postures) else "  [SKIP ]"
        print(f"{mark} {posture:20s}: {used_files:3d} / {total_files:3d}")

    print("\n================ 特征输入配置 ================")
    print(f"TRAIN_FEATURE_KEYS: {feat_cfg.feature_keys}")
    print(f"是否包含 ToF: {'tof_gated' in model_feature_names or 'tof_use' in model_feature_names}")
    print(f"输入维度 D: {len(model_feature_names)}")
    print("输入特征顺序:")
    for i, n in enumerate(model_feature_names):
        print(f"  x[{i:2d}] = {n}")

    if ENABLE_FEATURE_WEIGHTING:
        print("\n[Feature Weighting] enabled (after StandardScaler):")
        print(f"  tof_gated_weight = {TOF_GATED_WEIGHT}")
        print(f"  tof_use_weight   = {TOF_USE_WEIGHT}")

    if not args.no_confirm:
        if not ask_yes_no("\n确认以上数据无误,是否继续开始训练？", default=True):
            print("你选择了停止:脚本将退出(未进行训练/导出).")
            return

    (
        X_train, y_train, g_train,
        X_val, y_val, g_val,
        X_test, y_test, g_test,
        split_method, split_debug,
    ) = build_dataset_splits(
        data_root=data_root,
        postures=selected_postures,
        max_files_per_posture=max_files,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=args.seed,
        feat_cfg=feat_cfg,
    )

    if len(X_val) > 0:
        X_all = np.vstack([X_train, X_val, X_test]).astype(np.float32)
        y_all = np.concatenate([y_train, y_val, y_test])
        groups_all = np.concatenate([g_train, g_val, g_test])
    else:
        X_all = np.vstack([X_train, X_test]).astype(np.float32)
        y_all = np.concatenate([y_train, y_test])
        groups_all = np.concatenate([g_train, g_test])

    print("\n================ 数据集概览 ================")
    print(f"Split method: {split_method}")
    if split_debug:
        print("Split debug (简要):")
        for k in ("split_mode", "participants_total", "participants_used", "train_samples", "val_samples", "test_samples"):
            if k in split_debug:
                print(f"  {k}: {split_debug.get(k)}")

    print(f"Train X shape: {X_train.shape} (N,D), unique participants: {len(np.unique(g_train))}")
    print(f"Val   X shape: {X_val.shape} (N,D), unique participants: {len(np.unique(g_val))}")
    print(f"Test  X shape: {X_test.shape} (N,D), unique participants: {len(np.unique(g_test))}")
    print(f"All   X shape: {X_all.shape} (N,D), unique participants: {len(np.unique(groups_all))}")

    print("\n每个 set 的 posture 样本数量:")
    for set_name, yy in (("train", y_train), ("val", y_val), ("test", y_test)):
        if len(yy) == 0:
            print(f"  [{set_name}] (empty)")
            continue
        uniq, cnt = np.unique(yy, return_counts=True)
        s = ", ".join([f"{u}={int(c)}" for u, c in zip(uniq, cnt)])
        print(f"  [{set_name}] {s}")

    hl_str = str(args.ann_hidden_layers).strip()
    if hl_str:
        hidden_layers = [int(s) for s in hl_str.split(",") if s.strip()]
    else:
        hidden_layers = []

    model_cfg = ModelConfig(
        hidden_layers=hidden_layers,
        activation=str(args.ann_activation),
        dropout=float(args.ann_dropout),

        enable_fsr_sensor_masking=bool(ENABLE_FSR_SENSOR_MASKING),
        fsr_sensor_mask_prob=float(FSR_SENSOR_MASK_PROB),
        enable_fsr_domain_randomization=bool(ENABLE_FSR_DOMAIN_RANDOMIZATION),
        fsr_dr_scale_range=tuple(FSR_DR_SCALE_RANGE),
        fsr_dr_noise_std=float(FSR_DR_NOISE_STD),
        fsr_dr_bias_std=float(FSR_DR_BIAS_STD),

        enable_feature_weighting=bool(ENABLE_FEATURE_WEIGHTING),
        tof_gated_weight=float(TOF_GATED_WEIGHT),
        tof_use_weight=float(TOF_USE_WEIGHT),

        batch_size=int(args.ann_batch_size),
        epochs=int(args.ann_epochs),
        lr=float(args.ann_lr),
        weight_decay=float(args.ann_weight_decay),

        lr_scheduler=str(args.ann_lr_scheduler),
        lr_min=float(args.ann_lr_min),
        lr_plateau_factor=float(args.ann_lr_plateau_factor),
        lr_plateau_patience=int(args.ann_lr_plateau_patience),
        lr_plateau_threshold=float(args.ann_lr_plateau_threshold),
        lr_plateau_cooldown=int(args.ann_lr_plateau_cooldown),

        wd_scheduler=str(args.ann_wd_scheduler),
        weight_decay_end=float(args.ann_weight_decay_end),

        early_stop_patience=int(args.ann_early_stop_patience),
        min_delta=float(args.ann_min_delta),
        use_class_weights=bool(int(args.ann_use_class_weights)),

        device=str(args.ann_device),
        verbose=int(args.ann_verbose),
        random_state=int(args.seed),
    )

    cv_summary = None
    if ENABLE_CROSS_VALIDATION:
        cv_summary = run_group_cross_validation(
            feature_names=model_feature_names,
            X=X_all,
            y=y_all,
            groups=groups_all,
            model_cfg=model_cfg,
            seed=int(args.seed),
            method=str(CROSS_VALIDATION_METHOD),
            n_splits=int(CROSS_VALIDATION_N_SPLITS),
            out_dir=(out_dir if ENABLE_MODEL_EXPORT else None),
        )

    pipeline, metrics, _ = train_and_evaluate(
        feature_names=model_feature_names,
        X_train=X_train, y_train=y_train, g_train=g_train,
        X_val=X_val, y_val=y_val, g_val=g_val,
        X_test=X_test, y_test=y_test, g_test=g_test,
        model_cfg=model_cfg,
        seed=args.seed,
        split_method=split_method,
        split_debug=split_debug,
    )

    print("\n================ 评估结果 ================")
    ann = metrics.get("ann", {})
    print("Model: ANN(MLP)")
    print(f"ANN hidden_layers: {ann.get('hidden_layers')}")
    print(f"ANN activation: {ann.get('activation')}, dropout: {ann.get('dropout')}")
    print(f"Feature weighting: {ann.get('feature_weighting')}  indices={ann.get('feature_weight_indices')}  values={ann.get('feature_weight_values')}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report_text"])

    out_dir = Path(OUTPUT_ROOT_PATH).expanduser().resolve()
    cm_path = out_dir / OUT_CM_NAME
    header_path = out_dir / OUT_HEADER_NAME
    config_path = out_dir / OUT_CONFIG_NAME
    bundle_path = out_dir / OUT_BUNDLE_NAME

    cm = metrics["confusion_matrix"]
    labels = metrics["labels"]
    test_counts = metrics["test_counts"]

    test_error = 1.0 - float(metrics.get("accuracy", 0.0))
    print(f"Test error: {test_error:.4f}")

    if not ENABLE_MODEL_EXPORT:
        try:
            import pandas as _pd
            cm_df = _pd.DataFrame(cm, index=labels, columns=labels)
            print("\nConfusion matrix (rows=True label, cols=Pred label):")
            print(cm_df)
        except Exception:
            print("\nConfusion matrix (rows=True label, cols=Pred label):")
            print(cm)

        print("\n================ 完成 ================")
        print("已关闭模型导出:未生成任何模型文件(头文件/config/CM.jpg/bundle).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_standard(cm=cm, labels=labels, test_counts=test_counts, out_path=cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    class_names = list(pipeline.named_steps["clf"].classes_)
    export_model_to_c_header(pipeline, model_feature_names, class_names, header_path)
    print(f"C header saved to: {header_path}")

    config = {
        "input_features": model_feature_names,
        "input_dim": int(len(model_feature_names)),
        "classes": class_names,
        "feature_building": asdict(feat_cfg),
        "model": {
            "type": "ann",
            "hidden_layers": list(model_cfg.hidden_layers),
            "activation": model_cfg.activation,
            "dropout": model_cfg.dropout,

            "feature_weighting": bool(model_cfg.enable_feature_weighting),
            "tof_gated_weight": float(model_cfg.tof_gated_weight),
            "tof_use_weight": float(model_cfg.tof_use_weight),

            "fsr_sensor_masking": bool(model_cfg.enable_fsr_sensor_masking),
            "fsr_sensor_mask_prob": float(model_cfg.fsr_sensor_mask_prob),
            "fsr_domain_randomization": bool(model_cfg.enable_fsr_domain_randomization),
            "fsr_dr_scale_range": [float(model_cfg.fsr_dr_scale_range[0]), float(model_cfg.fsr_dr_scale_range[1])],
            "fsr_dr_noise_std": float(model_cfg.fsr_dr_noise_std),
            "fsr_dr_bias_std": float(model_cfg.fsr_dr_bias_std),

            "batch_size": model_cfg.batch_size,
            "epochs_config": model_cfg.epochs,
            "epochs_trained": metrics.get("ann", {}).get("epochs_trained"),
            "best_epoch": metrics.get("ann", {}).get("best_epoch"),
            "best_val_loss": metrics.get("ann", {}).get("best_val_loss"),

            "lr": model_cfg.lr,
            "weight_decay": model_cfg.weight_decay,
            "lr_scheduler": model_cfg.lr_scheduler,
            "lr_min": model_cfg.lr_min,
            "lr_plateau_factor": model_cfg.lr_plateau_factor,
            "lr_plateau_patience": model_cfg.lr_plateau_patience,
            "lr_plateau_threshold": model_cfg.lr_plateau_threshold,
            "lr_plateau_cooldown": model_cfg.lr_plateau_cooldown,

            "wd_scheduler": model_cfg.wd_scheduler,
            "weight_decay_end": model_cfg.weight_decay_end,

            "early_stop_patience": model_cfg.early_stop_patience,
            "min_delta": model_cfg.min_delta,
            "use_class_weights": model_cfg.use_class_weights,
            "device": model_cfg.device,
            "random_state": model_cfg.random_state,
        },
        "dataset": {
            "data_root": str(data_root),
            "output_root": str(out_dir),
            "postures": list(selected_postures),
            "max_files_per_posture": max_files,
            "rows_per_posture": rows_per_posture,
            "rows_per_file": rows_per_file,
            "X_shape": list(X_all.shape),
            "X_train_shape": list(X_train.shape),
            "X_val_shape": list(X_val.shape),
            "X_test_shape": list(X_test.shape),
            "seed": args.seed,
            "split_method": metrics["split_method"],
            "split_debug": metrics.get("split_debug", {}),
            "group_cnt_per_class": metrics["group_cnt_per_class"],
            "unique_groups": int(len(np.unique(groups_all))),
            "unique_groups_train": int(len(np.unique(g_train))),
            "unique_groups_val": int(len(np.unique(g_val))),
            "unique_groups_test": int(len(np.unique(g_test))),
        },
        "outputs": {
            "header": OUT_HEADER_NAME,
            "config": OUT_CONFIG_NAME,
            "confusion_matrix": OUT_CM_NAME,
            "bundle": OUT_BUNDLE_NAME,
        },
    }

    if cv_summary is not None:
        config["cross_validation"] = cv_summary
        config.setdefault("outputs", {})["cv_summary"] = OUT_CV_SUMMARY_NAME
        config.setdefault("outputs", {})["cv_confusion_matrix"] = OUT_CV_CM_NAME

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config saved to: {config_path}")

    bundle: Dict[str, object] = {
        "model_type": "ann",
        "classes": class_names,
        "feature_building": asdict(feat_cfg),
        "input_features": model_feature_names,
        "ann_config": asdict(model_cfg),
    }

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    bundle["scaler_mean"] = scaler.mean_.astype(np.float32)
    bundle["scaler_scale"] = scaler.scale_.astype(np.float32)

    clf: TorchMLPClassifier = pipeline.named_steps["clf"]
    if clf.model_ is None:
        raise RuntimeError("bundle 导出失败:ANN 模型未训练(clf.model_ is None).")

    linear_layers: List[nn.Linear] = [m for m in clf.model_ if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("bundle 导出失败:未在 ANN 模型中找到 Linear 层.")

    bundle["ann_layer_sizes"] = [int(len(scaler.mean_))] + [int(x) for x in clf.hidden_layers] + [int(len(class_names))]
    bundle["ann_activation"] = str(clf.activation)
    bundle["ann_weights"] = [m.weight.detach().cpu().numpy().astype(np.float32) for m in linear_layers]
    bundle["ann_biases"] = [m.bias.detach().cpu().numpy().astype(np.float32) for m in linear_layers]
    bundle["ann_epochs_trained"] = int(clf.epochs_trained_)
    bundle["ann_best_epoch"] = None if clf.best_epoch_ is None else int(clf.best_epoch_)
    bundle["ann_best_val_loss"] = None if clf.best_val_loss_ is None else float(clf.best_val_loss_)

    joblib.dump(bundle, bundle_path)
    print(f"Model bundle saved to: {bundle_path}")

    print("\n================ 完成 ================")
    print("输出文件:")
    print(f"  {header_path}")
    print(f"  {config_path}")
    print(f"  {cm_path}")
    print(f"  {bundle_path}")


if __name__ == "__main__":
    main()
