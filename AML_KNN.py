# -*- coding: utf-8 -*-
"""
AML_KNN.py  (Updated: ToF 门控在采集端完成)

====================================================================
本脚本用途(PC 端训练 + 导出嵌入式可用模型):
--------------------------------------------------------------------
1) 从坐姿 CSV/Excel(导出为 CSV) 数据中读取并整理"工程特征"(不额外生成新列),作为 KNN 的输入.
2) 在 PC 上训练 KNN 分类器(支持 group-aware CV 超参搜索):
   - 邻居数 n_neighbors
   - 距离权重 weights(uniform/distance)
   - 距离度量 metric/p (minkowski/euclidean/manhattan)
3) 在测试集上评估(分类报告、混淆矩阵).
4) 导出为 Arduino/ESP32 可直接编译使用的 C 头文件:AML_KNN.h

====================================================================
你论文中的数据文件结构(本脚本只支持这一种):
--------------------------------------------------------------------
AMLDEVICE/
  Straight/
    participant_1.csv
    participant_2.csv
  Left/
    participant_1.csv
  Right/
    participant_1.csv
  ...

说明:
- "文件夹名" = posture 类别名(标签)
- "每个 participant_x.csv" = 1 个参与者(同名 participant 视为同一个人)
- 训练/验证/测试切分:
  - participant 数量 < 3 : 按样本随机划分(0.7/0.15/0.15)
  - participant 数量 >=3 : 按 participant 指定划分(你在全局变量区用数字指定哪些人属于 train/val/test)

====================================================================
重要变更:ToF 门控(tof_gated/tof_use)在采集端完成
--------------------------------------------------------------------
你已经在 Arduino/ESP32 采集程序中直接计算并写入 CSV/Excel:
- tof_gated : 门控后的 ToF 值(不可用/侧倾时用 0.5 填充)
- tof_use   : ToF 是否启用(0/1)

因此:
- 本脚本[不再从 tof_norm 计算 tof_gated/tof_use](按你的要求删除)
- 若你在 SELECTED_FEATURE_KEYS 中选择了 "TOF",则 CSV 必须包含 tof_gated、tof_use 两列,否则直接报错

====================================================================
新增/修复点(对应你截图里提到的问题):
--------------------------------------------------------------------
[修复 1] group 采用"参与者 ID"(从文件名 participant_1 提取),而不是文件相对路径
        -> 避免同一个参与者在不同 posture 文件夹里被当成不同 group 导致数据泄漏

[修复 2] 不再对整个 DataFrame 做 dropna(how="any")
        -> 只对"最终特征矩阵 X"做有限清洗:去掉 NaN/Inf 行,避免额外列导致误删

[新增] 训练/验证/测试切分:支持按 participant 指定划分 train/val/test (以及 participant<3 的样本级划分)
        -> participant 数量 >=3: 由 TRAIN/VALIDATION/TEST_PARTICIPANT_NUMBERS 指定 participant 归属
        -> participant 数量 <3 : 读取完所有样本后按 0.7/0.15/0.15 随机划分(余数给 train)

[修复 4] 特征顺序对齐 CSV 列顺序
        -> 让导出的 AML_KNN.h 中 x[i] 顺序与 CSV 一致,降低 MCU 端填错顺序的风险

[修复 5] max_files_per_posture 不再按文件名排序截断,而是每个 posture 内随机抽取 participant 文件
        -> 避免快速实验时总是偏向 participant_1..N
        -> 同时在 posture 内合并后会再随机打乱样本顺序

[优化] CSV 读取:先读表头再用 C engine + list usecols 读取,避免 engine="python" 的性能损失
[优化] 可选 rows_per_file:每个 CSV 文件固定抽样 K 行,避免某个文件过长导致"某参与者权重过大"

====================================================================
你的 CSV 关键列(示例 participant_1.csv)
--------------------------------------------------------------------
至少需要包含以下列(本脚本会用到):
- fsr0_norm..fsr9_norm
- lr_ratio, ud_ratio
- cop_x_ratio, cop_y_ratio
- (若选择了 "TOF",还需要)tof_gated, tof_use

可选列(不会影响训练,仅作为数据记录/调试):
- tof_norm
- fsr*_std
- t_ms
- 其它任何列

本脚本不会在 PC 侧补齐/推导任何新的特征列(你要求特征由采集端/预处理端写入 CSV).
只会对读取到的特征做 NaN/Inf 清洗(不再做额外裁剪/归一化)。

注意:
- 你明确要求:lr_ratio/ud_ratio 已经在 CSV 里有,因此这里不再用 FSR 重新计算它们.
  如果 CSV 缺少 lr_ratio/ud_ratio,本脚本会直接报错提示缺列.

====================================================================
输入维度(ToF 特征开关)
--------------------------------------------------------------------
由 import 后的 SELECTED_FEATURE_KEYS 决定。

示例:
- SELECTED_FEATURE_KEYS 包含 "FSR" + "TOF" + 其它特征 -> 输入维度=16
- SELECTED_FEATURE_KEYS 仅包含 "FSR" + 其它特征(不含 "TOF") -> 输入维度=14

====================================================================
输出文件(固定文件名):
--------------------------------------------------------------------
输出目录(固定在代码中,不允许命令行覆盖)下会生成:
  AML_KNN.h                 -> 推理头文件(KNN)
  AML_KNN_config.json       -> 配置与复现信息
  AML_KNN_CM.jpg            -> 混淆矩阵图片
  AML_KNN_bundle.joblib     -> PC 侧调试用的参数包
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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# ★★ 所有可调参数(请只在这里修改默认值) ★★
#    你要求:把所有可调参数放到 import 之后的显眼位置
# ============================================================

AML_OS = "MAC"  # "WIN" / "MAC"

if AML_OS.strip().upper() == "WIN":
    # Windows:保持你当前电脑的固定路径
    DATA_ROOT_PATH = r"C:\Users\17717\Desktop\AML_NEW_DATA"
    OUTPUT_ROOT_PATH = r"C:\Users\17717\Desktop\AML_MODEL\KNN"
elif AML_OS.strip().upper() == "MAC":
    # macOS:桌面固定目录(若 Desktop 不存在则回退到用户主目录)
    _desktop = Path.home() / "Desktop"
    if not _desktop.exists():
        _desktop = Path.home()
    DATA_ROOT_PATH = str(_desktop / "AML_CODE"/ "AML_DATASET")
    OUTPUT_ROOT_PATH = str(_desktop / "AML_MODEL" / "KNN")
else:
    raise ValueError("AML_OS 只能是 'WIN' 或 'MAC'.")


# ---------- 固定输出文件名(按你指定) ----------
OUT_HEADER_NAME = "AML_KNN.h"
OUT_CONFIG_NAME = "AML_KNN_config.json"
OUT_CM_NAME = "AML_KNN_CM.jpg"
OUT_BUNDLE_NAME = "AML_KNN_bundle.joblib"

# ---------- 输出开关 ----------
# True : 训练完成后把模型/配置/混淆矩阵等写入 OUTPUT_ROOT_PATH 指定目录
# False: 不写任何文件,只在终端打印 test error / confusion matrix 等评估指标
EXPORT_MODEL_FILES = True

# ---------- 默认训练数据采样参数 ----------
DEFAULT_MAX_FILES_PER_POSTURE = 0   # 0=不限制;>0=每类最多使用多少个 participant 文件
DEFAULT_ROWS_PER_POSTURE = 0        # 0=不限制;>0=每类最多抽样多少行(在 posture 合并后抽)
DEFAULT_ROWS_PER_FILE = 0           # 0=不限制;>0=每个 CSV 文件固定抽样多少行(先抽,再合并)
DEFAULT_SEED = 42
DEFAULT_NO_CONFIRM = False

# ---------- 新增: Train/Validation/Test 划分(按 participant 指定) ----------
# 说明(按你的最新要求):
# - participant 数量 < 3 : 继续使用样本级随机划分(0.7/0.15/0.15),这里不要改,也不会读取下面的指定列表。
# - participant 数量 >=3: 你可以在这里用“数字”指定每个数据集包含哪些 participant。
#   例如:
#     TRAIN_PARTICIPANT_NUMBERS = [1, 2]
#     TEST_PARTICIPANT_NUMBERS  = [3]
#   对应 participant_1、participant_2 进入训练集,participant_3 进入测试集。
# - 约束:
#   - train 与 test 的 participant 数量不能为 0(否则脚本报错并停止训练)
#   - validation 允许为空([])
#   - 同一个 participant 不能同时出现在多个集合(否则报错)

# 你只需要填“数字”,不要写 participant_ 前缀。
TRAIN_PARTICIPANT_NUMBERS: List[Union[int, str]] = [1, 2, 3, 5, 6]
VALIDATION_PARTICIPANT_NUMBERS: List[Union[int, str]] = [7]
TEST_PARTICIPANT_NUMBERS: List[Union[int, str]] = [4]

# ---------- 训练输入特征选择(只需要在这里改) ----------
# 你可以在这里用一个列表,控制本次训练/导出模型要使用哪些特征.
# 可选项:
#   "FSR" -> 10 个 fsr0_norm..fsr9_norm
#   "TOF" -> 2 个 tof_gated, tof_use
#   "lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio" -> 单个特征(各 1 维)
#
# 示例:
#   只训练 FSR + TOF:
#     ["FSR", "TOF"]
#   训练全部特征:
#     ["FSR", "TOF", "lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio"]
#
# 注意:
# - 最终喂给模型的顺序固定为:
#     (若包含)FSR(0..9) -> (若包含)TOF(tof_gated,tof_use) -> (若包含)lr/ud/cop_x/cop_y
# - CSV 必须包含你选择的所有列;缺列会直接报错.
SELECTED_FEATURE_KEYS: List[str] = ["FSR", "TOF", "lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio"] 

# ---------- Feature Dropout / Sensor Masking (训练增强:提升泛化) ----------
# 对 FSR 的每个维度,以概率 p 置 0,模拟个别传感器失效/接触不良/读数丢失.
ENABLE_FSR_SENSOR_MASKING = False
FSR_SENSOR_MASK_PROB = 0.0  # 每个 FSR 维度被置 0 的概率(0~1). 建议 0.05~0.30

# ---------- Domain Randomization (FSR, 训练增强:提升跨人/跨设备泛化) ----------
# 对每个样本的 FSR(10 维)做随机扰动,模拟传感器增益差异与噪声.
ENABLE_FSR_DOMAIN_RANDOMIZATION = False
FSR_DR_SCALE_RANGE = (0.8, 1.2)  # 对每个 FSR 维度做乘法扰动: Uniform(low, high)
FSR_DR_NOISE_STD = 0.05          # 对每个 FSR 维度加性噪声: Normal(0, std)
FSR_DR_BIAS_STD = 0.0            # 可选:对整条 FSR 向量加同一个 bias: Normal(0, std). 0=关闭(建议 0.00~0.05)


# ---------- 新增: Cross Validation 开关(用于超参搜索) ----------
# 说明:
# - True : 在超参搜索阶段使用 group-aware CV (GroupKFold/LeaveOneGroupOut, groups=participant)
# - False: 不做 CV 搜索,直接使用你通过命令行传入的超参(C/gamma/degree/coef0)
ENABLE_CROSS_VALIDATION = False

# 选择 CV 策略:
# - "auto" : group 数量<=CV_LOGO_MAX_GROUPS 用 LeaveOneGroupOut,否则用 GroupKFold
# - "logo" : 强制 LeaveOneGroupOut
# - "groupkfold" : 强制 GroupKFold
CROSS_VALIDATION_STRATEGY = "logo"  # "auto" | "logo" | "groupkfold"

# 仅对策略 auto/groupkfold 生效
CV_GROUPKFOLD_N_SPLITS = 5
CV_LOGO_MAX_GROUPS = 10

# ---------- group 解析规则(从文件名提取 participant ID) ----------
DEFAULT_PARTICIPANT_REGEX = r"(participant[_-]?\d+)"   # 如 participant_1 / participant-1
DEFAULT_GROUP_LOWERCASE = True                         # group_id 是否统一转小写(建议 True 保持一致)

# ---------- 默认模型参数(KNN) ----------
# 说明:
# - KNN 原生支持多分类.
# - 仍保留 group-aware CV 做超参搜索,与现有流程一致.
# - KNN 的嵌入式体积/耗时主要受训练样本数量影响,可用 max_export_samples 控制.

# (1) 最终训练默认超参
KNN_N_NEIGHBORS = 7
KNN_WEIGHTS = "distance"          # "uniform" | "distance"
KNN_METRIC = "minkowski"          # "minkowski" | "euclidean" | "manhattan"
KNN_P = 1                         # 仅 metric=minkowski 时生效
KNN_ALGORITHM = "auto"            # "auto" | "ball_tree" | "kd_tree" | "brute"
KNN_LEAF_SIZE = 20
KNN_N_JOBS = -1                    # 1=单线程更稳; 需要并行可改为 -1

# 导出约束(仅影响导出前检查;0=不限制)
KNN_MAX_EXPORT_SAMPLES = 0

# (2) KNN 超参搜索空间(ENABLE_CROSS_VALIDATION=True 时生效)
KNN_SEARCH_N_NEIGHBORS_LIST: List[int] = [1, 3, 5, 7, 9, 11]
KNN_SEARCH_WEIGHTS_LIST: List[str] = ["uniform", "distance"]
KNN_SEARCH_METRIC_LIST: List[str] = ["minkowski", "euclidean", "manhattan"]
KNN_SEARCH_P_LIST: List[int] = [1, 2]
KNN_SEARCH_ALGORITHM_LIST: List[str] = ["auto", "brute"]
KNN_SEARCH_LEAF_SIZE_LIST: List[int] = [20, 30, 40]

# (3) 搜索次数上限:当候选组合太多时,随机抽样最多 K 组(保证可控运行时间)
KNN_SEARCH_MAX_TRIALS = 500
# ---------- 训练前禁用的命令行参数(你要求删除覆盖功能) ----------
FORBIDDEN_CLI_ARGS = ("--data_root", "--output_dir")

# ---------- Feature Weighting (提高某些特征重要性; 训练+导出推理保持一致) ----------
# 原理：在 RobustScaler 之后对指定特征乘权重 w>1
ENABLE_FEATURE_WEIGHTING = True

# 只加权 tof_gated（你要的）
TOF_GATED_WEIGHT = 3.0  # 建议 1.2~3.0 之间试


# ============================================================
# 1) 特征列名与顺序
# ============================================================

# 10 个 FSR 原始输入(与 CSV 列一致)
FSR_FEATURE_NAMES: List[str] = [f"fsr{i}_norm" for i in range(10)]

# 其它输入特征(CSV 必须直接提供)
# 注意：这里的顺序就是最终喂给模型的顺序(在 FSR/ToF 之后).
# 为降低 MCU 端把特征顺序填错的风险,这里刻意对齐 CSV 列的原始顺序(lr/ud/cop_x/cop_y).
FSR_DERIVED_FEATURE_NAMES: List[str] = [
    "lr_ratio",
    "ud_ratio",
    "cop_x_ratio",
    "cop_y_ratio",
]

# ToF 相关特征(由采集端输出到 CSV)
TOF_FEATURE_NAMES: List[str] = [
    "tof_gated",
    "tof_use",
]




class FsrAugmenter(BaseEstimator, TransformerMixin):
    """
    仅在训练阶段对 FSR 特征做增强,用于提升跨人/跨设备泛化.

    支持两类增强(均只作用在 FSR(10 维)上,其它特征保持不变):
    1) Domain Randomization:
       - per-dim scale:  Uniform(low, high)
       - per-dim noise:  Normal(0, noise_std)
       - optional bias:  per-sample Normal(0, bias_std),并加到整条 FSR 向量(同一个 bias)
    2) Sensor Masking / Feature Dropout:
       - 对每个 FSR 维度,以概率 p 置 0,模拟个别传感器失效/接触不良/读数丢失.

    关键设计:
    - 在 sklearn Pipeline 中使用: fit_transform() 对训练数据做增强;
      transform() 对验证/测试数据不做增强,从而避免评估集分布被污染.
    """

    def __init__(
        self,
        enable_sensor_masking: bool = True,
        mask_prob: float = 0.0,
        enable_domain_randomization: bool = True,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        bias_std: float = 0.0,
        fsr_indices: Optional[Sequence[int]] = None,
        random_state: Optional[int] = None,
    ):
        self.enable_sensor_masking = bool(enable_sensor_masking)
        self.mask_prob = float(mask_prob) if mask_prob is not None else 0.0
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.scale_range = tuple(scale_range) if scale_range is not None else (1.0, 1.0)
        self.noise_std = float(noise_std) if noise_std is not None else 0.0
        self.bias_std = float(bias_std) if bias_std is not None else 0.0
        self.fsr_indices = list(fsr_indices) if fsr_indices is not None else []
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 推理/评估阶段不做增强,只保证 dtype 一致
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        X = np.asarray(X, dtype=np.float32)

        # 只处理 (N, D)
        if X.ndim != 2 or X.shape[0] == 0:
            return X
        if not self.fsr_indices:
            return X

        X_aug = X.copy()
        cols = np.array(self.fsr_indices, dtype=int)
        fsr = X_aug[:, cols]  # view:修改 fsr 即修改 X_aug

        rng = np.random.default_rng(self.random_state)

        # 1) Domain Randomization
        if self.enable_domain_randomization:
            try:
                low, high = float(self.scale_range[0]), float(self.scale_range[1])
            except Exception:
                low, high = 1.0, 1.0
            if high < low:
                low, high = high, low

            if (low != 1.0) or (high != 1.0):
                scale = rng.uniform(low, high, size=fsr.shape).astype(np.float32)
                fsr *= scale

            if self.noise_std > 0.0:
                noise = rng.normal(0.0, self.noise_std, size=fsr.shape).astype(np.float32)
                fsr += noise

            if self.bias_std > 0.0:
                bias = rng.normal(0.0, self.bias_std, size=(fsr.shape[0], 1)).astype(np.float32)
                fsr += bias

        # 2) Sensor Masking (放到最后,保证被 mask 的维度最终为 0)
        if self.enable_sensor_masking:
            p = float(self.mask_prob) if self.mask_prob is not None else 0.0
            if p > 0.0:
                p = min(max(p, 0.0), 1.0)
                mask = rng.random(size=fsr.shape) < p
                fsr[mask] = 0.0

        return X_aug




# ============================================================
# 2) 小工具函数(排序/交互/group 解析)
# ============================================================

def natural_sort_key(s: str):
    """自然排序 key:让 participant_2 排在 participant_10 前面."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """y/n 交互开关(用于"中断/继续"确认)."""
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
    """
    从文件名中提取参与者 ID,用作 group.

    你的规则:同名 participant 是一个人.
    例如:
      Straight/participant_1.csv
      Left/participant_1.csv
    都应该属于同一个 group:participant_1

    若未匹配到 participant_x,则退化为 file_path.stem(不含扩展名).
    """
    stem = file_path.stem
    m = re.search(DEFAULT_PARTICIPANT_REGEX, stem, flags=re.IGNORECASE)
    gid = m.group(1) if m else stem
    return gid.lower() if DEFAULT_GROUP_LOWERCASE else gid


# ============================================================
# 3) 目录扫描:只支持 "子文件夹=posture,文件=participant"
# ============================================================

def find_posture_sources_folder_only(data_root: Path, posture: str) -> List[Path]:
    """找到 posture 子文件夹下所有 participant 的 csv 文件(自然排序)."""
    posture_dir = data_root / posture
    if not posture_dir.exists() or not posture_dir.is_dir():
        return []
    return sorted(posture_dir.glob("*.csv"), key=lambda p: natural_sort_key(p.name))


def auto_detect_postures_folder_only(data_root: Path) -> List[str]:
    """扫描 data_root 下所有 posture 子文件夹(只认子文件夹,不认根目录文件)."""
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
    """训练前扫描目录:展示 posture 列表及 participant 数量(used/total)."""
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
# 4) 读取 CSV 并构造特征(ToF 门控由采集端完成)
# ============================================================

@dataclass
class FeatureBuildConfig:
    """
    用于从 CSV 构造特征的配置(会写入 JSON 以便复现).

    注意:
    - 本脚本不再计算/补齐任何特征列,只读取 CSV 中已经存在的列
    - 训练时使用哪些特征由 feature_keys 控制(对应 import 后的 SELECTED_FEATURE_KEYS)
    - tof_gated/tof_use 由采集端计算,本脚本只读取(若 feature_keys 包含 "TOF")
    - lr_ratio/ud_ratio 等比值特征必须由 CSV 提供(若 feature_keys 选择了它们)
    """
    feature_keys: List[str] = field(default_factory=lambda: list(SELECTED_FEATURE_KEYS))

    def __post_init__(self):
        # 规范化/去重/校验,保证整个脚本内部只看到“规范化后的 key 列表”
        self.feature_keys = normalize_feature_keys(self.feature_keys)

def _require_cols(df: pd.DataFrame, cols: List[str], file_path: Path):
    """如果缺列,给出清晰报错(包含当前文件与现有列名)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列:{missing}\n"
            f"文件:{file_path}\n"
            f"现有列:{list(df.columns)}"
        )


def normalize_feature_keys(feature_keys: Optional[Sequence[str]]) -> List[str]:
    """
    把用户填写的 feature_keys 规范化为脚本内部使用的标准形式,并做校验/去重.

    允许的 key:
      - "FSR" : 10 个 fsr0_norm..fsr9_norm
      - "TOF" : tof_gated, tof_use
      - "lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio"
      - "ALL" : 等价于全部特征(便于快速切换)

    返回:
      规范化后的 key 列表,顺序固定为:
        (若包含)FSR -> (若包含)TOF -> (若包含)lr/ud/cop_x/cop_y
    """
    if feature_keys is None:
        feature_keys = []

    # 1) 字符串化 + 基础规范化
    raw: List[str] = []
    for k in feature_keys:
        if k is None:
            continue
        s = str(k).strip()
        if not s:
            continue

        su = s.upper()
        if su in ("ALL", "*", "ALL_FEATURES"):
            # 直接展开为全部特征
            return ["FSR", "TOF"] + list(FSR_DERIVED_FEATURE_NAMES)

        if su in ("FSR", "TOF"):
            raw.append(su)
        else:
            raw.append(s.lower())

    # 2) 校验
    allowed = {"FSR", "TOF"} | set(FSR_DERIVED_FEATURE_NAMES)
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise ValueError(
            "SELECTED_FEATURE_KEYS 中包含未知项: "
            f"{unknown}. 允许的选项: {sorted(allowed, key=natural_sort_key)}"
        )

    # 3) 固定输出顺序(保证训练/导出/MCU 端顺序一致)
    raw_set = set(raw)
    keys_out: List[str] = []
    if "FSR" in raw_set:
        keys_out.append("FSR")
    if "TOF" in raw_set:
        keys_out.append("TOF")
    for k in FSR_DERIVED_FEATURE_NAMES:
        if k in raw_set:
            keys_out.append(k)

    return keys_out


def get_model_feature_names(feature_keys: Optional[Sequence[str]]) -> List[str]:
    """根据 feature_keys 返回"最终喂给模型"的特征顺序."""
    keys = normalize_feature_keys(feature_keys)
    kset = set(keys)

    names: List[str] = []
    if "FSR" in kset:
        names += FSR_FEATURE_NAMES
    if "TOF" in kset:
        names += TOF_FEATURE_NAMES
    for n in FSR_DERIVED_FEATURE_NAMES:
        if n in kset:
            names.append(n)
    return names

def build_features_from_df(df: pd.DataFrame, cfg: FeatureBuildConfig, file_path: Path) -> np.ndarray:
    """
    把一个 CSV 的 DataFrame 变成 (N, D).

    重要约束(按你的要求):
    - 本脚本不再计算/补齐任何特征列
    - CSV 必须包含模型所需的全部特征列(缺列直接报错)
    - 不再使用 cop_x_centered / cop_y_centered / cop_radius
    - 直接使用 cop_x_ratio / cop_y_ratio 作为输入特征
    """
    model_features = get_model_feature_names(cfg.feature_keys)

    # 1) 训练输入必须在文件里齐全
    _require_cols(df, model_features, file_path)

    # 2) 直接读取最终输入特征
    X = df[model_features].to_numpy(dtype=np.float32)

    if X.shape[0] == 0:
        return np.zeros((0, len(model_features)), dtype=np.float32)

    # 3) 维度保护
    expected_dim = len(model_features)
    if X.shape[1] != expected_dim:
        raise RuntimeError(f"内部错误:特征维度不是 {expected_dim},而是 {X.shape[1]}")

    return X.astype(np.float32)



def load_one_excel(file_path: Path, feat_cfg: FeatureBuildConfig) -> np.ndarray:
    """
    读取一个 participant 的 CSV 文件并返回 (N, D).

    清洗策略(针对你截图里 dropna 的问题做了改进):
    1) 不对整个 DataFrame dropna(how="any"),避免额外列导致误删
    2) 只基于"最终特征矩阵 X"做有限清洗:
       - 去掉包含 NaN/Inf 的样本行
       - 去掉 10 个 FSR 全 0 的行(无人坐/seatOk=false)
    """
    # ------------------------------------------------------------
    # 只读取"最终模型需要的特征列" + 可选调试列(提高读取速度)
    # ------------------------------------------------------------
    model_features = get_model_feature_names(feat_cfg.feature_keys)

    wanted_cols = set(
        model_features
        + ["tof_norm", "t_ms"]                    # 可选列：仅用于记录/调试
        + [f"fsr{i}_std" for i in range(10)]      # 可选列：仅用于记录/调试
    )

    # ------------------------------------------------------------
    # 性能优化:先只读取表头(nrows=0),把 usecols 从 callable 变成 list,
    #          这样可以使用 pandas 的 C engine,避免 engine="python" 带来的明显变慢
    # ------------------------------------------------------------
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



    # 把 inf/-inf 变成 NaN,后面统一用 isfinite 过滤
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.shape[0] == 0:
        return np.zeros((0, len(get_model_feature_names(feat_cfg.feature_keys))), dtype=np.float32)

    # 构造/读取特征(D = 16 or 14)
    X = build_features_from_df(df, feat_cfg, file_path)  # (N, D)

    # 去掉 NaN/Inf 行(只针对最终特征矩阵)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if X.shape[0] == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    # 去掉 fsr 全 0 行(仅当选择了 FSR 特征;否则无法判断 seatOk)
    fsr_idx = [model_features.index(n) for n in FSR_FEATURE_NAMES if n in model_features]
    if fsr_idx:
        fsr_sum = np.sum(X[:, fsr_idx], axis=1)
        X = X[fsr_sum > 1e-6]

    return X.astype(np.float32)


def _pick_reference_posture(data_root: Path, postures: Sequence[str]) -> str:
    """
    选择一个 posture 作为 "participant 数量/ID" 的参考.

    你要求:所有 posture 的 participant 数量一致,因此可以直接用 Straight 文件夹.
    若 Straight 不存在,则退化为 postures[0].
    """
    if (data_root / "Straight").exists():
        return "Straight"
    if not postures:
        raise RuntimeError("postures 为空:无法构建数据集.")
    return str(postures[0])


def _get_participant_ids_from_posture(data_root: Path, posture: str) -> List[str]:
    """从某个 posture 文件夹提取 participant(group) 列表(自然排序)."""
    files = find_posture_sources_folder_only(data_root, posture)
    if not files:
        raise FileNotFoundError(
            f"未找到参考 posture={posture} 的数据文件.\n"
            f"请检查:{data_root / posture} 下是否包含 participant_*.csv"
        )

    gids = [extract_participant_group_id(f) for f in files]
    # 保护:同一个 posture 下不应该出现重复的 participant id
    if len(gids) != len(set(gids)):
        dup = sorted({g for g in gids if gids.count(g) > 1})
        raise RuntimeError(
            f"参考 posture={posture} 下出现重复 participant id:{dup}\n"
            f"请检查文件命名是否冲突(例如 participant_1.csv 重复)."
        )
    return gids


def _extract_participant_number(participant_id: str) -> Optional[int]:
    """从 participant_id 末尾提取数字编号(例如 participant_3 -> 3)."""
    m = re.search(r"(\d+)$", str(participant_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _build_participant_number_map(participant_ids: Sequence[str]) -> Dict[int, str]:
    """
    构建:编号(int) -> participant_id(str) 映射.

    目的:
    - 让你在全局变量区只写数字(如 1,2,3),脚本能匹配到实际文件名里的 participant_x.
    - 同时兼容 participant_01 / participant_1 这类“前导 0”差异(若同时存在会报错,避免歧义).
    """
    mapping: Dict[int, str] = {}
    for pid in participant_ids:
        num = _extract_participant_number(pid)
        if num is None:
            raise RuntimeError(f"无法从 participant id 解析编号:{pid}")
        if num in mapping and mapping[num] != pid:
            raise RuntimeError(
                "检测到重复/歧义 participant 编号,同一个编号对应多个文件名.\n"
                f"number={num}, ids=[{mapping[num]}, {pid}]\n"
                "请统一 participant 文件命名(建议全部使用 participant_1, participant_2 ...)."
            )
        mapping[num] = pid
    return mapping


def _normalize_user_participants(
    user_numbers: Sequence[Union[int, str]],
    number_map: Dict[int, str],
    set_name: str,
) -> List[str]:
    """把用户填写的数字列表映射为 participant_id 列表,并做基本校验."""
    ids: List[str] = []
    seen_nums: set[int] = set()

    for raw in user_numbers:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw == "":
                continue
        try:
            num = int(raw)
        except Exception as e:
            raise RuntimeError(f"{set_name} 集合的 participant 编号必须是数字,但收到:{raw!r}") from e

        if num in seen_nums:
            raise RuntimeError(f"{set_name} 集合中出现重复 participant 编号:{num}")
        seen_nums.add(num)

        if num not in number_map:
            known = sorted(number_map.keys())
            raise RuntimeError(
                f"{set_name} 集合中指定了不存在的 participant 编号:{num}.\n"
                f"当前数据集中可用编号: {known}"
            )
        ids.append(number_map[num])

    return ids


def _split_counts_by_ratio(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """按比例切分样本数量:val/test 向下取整,余数全部给 train(按你要求)."""
    if n < 0:
        raise ValueError("n 必须 >=0")
    n_val = int(n * float(val_ratio))
    n_test = int(n * float(test_ratio))
    n_train = int(n) - n_val - n_test
    if n_train < 0:
        # 理论不会发生(因为 val/test 是 floor)
        n_train = 0
    return n_train, n_val, n_test


def build_dataset_splits(
    data_root: Path,
    postures: Sequence[str],
    max_files_per_posture: Optional[int],
    rows_per_posture: Optional[int],
    rows_per_file: Optional[int],
    seed: int,
    feat_cfg: FeatureBuildConfig,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # train
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # val
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # test
    str,                                        # split_method
    Dict[str, object],                          # split_debug
]:
    """
    构建数据集:返回 train/val/test 三份:
      (X_train, y_train, groups_train),
      (X_val,   y_val,   groups_val),
      (X_test,  y_test,  groups_test),
      split_method,
      split_debug

    切分规则(按你要求):
    - 若 participant 数量 <3:
        读取完该 posture 的所有 participant 后,随机打乱所有样本,
        然后按比例 0.7/0.15/0.15 切分到 train/val/test.
        不能整除的剩余样本全部给 train.
    - 若 participant 数量 >=3:
        按 "participant 指定" 切分:
        你在全局变量区用数字指定 train/val/test 的 participant 编号。
        (示例: TRAIN_PARTICIPANT_NUMBERS=[1,2], TEST_PARTICIPANT_NUMBERS=[3])
        约束:
        - train/test 的 participant 数量不能为 0(否则报错不训练)
        - validation 可以为空([])
        - 同一 participant 不能同时属于多个 set(否则报错)
    """
    rng = np.random.default_rng(seed)

    D = len(get_model_feature_names(feat_cfg.feature_keys))

    # ----------------------------
    # 1) participant 数量(用 Straight 作为参考,按你要求)
    # ----------------------------
    ref_posture = _pick_reference_posture(data_root, postures)
    ref_participant_ids = _get_participant_ids_from_posture(data_root, ref_posture)
    participant_total = int(len(ref_participant_ids))

    # participant_total>=3 才启用按 participant 指定划分; 否则退化为样本级切分
    use_participant_split = participant_total >= 3

    if use_participant_split:
        number_map = _build_participant_number_map(ref_participant_ids)

        train_ids = _normalize_user_participants(TRAIN_PARTICIPANT_NUMBERS, number_map, "train")
        val_ids = _normalize_user_participants(VALIDATION_PARTICIPANT_NUMBERS, number_map, "validation")
        test_ids = _normalize_user_participants(TEST_PARTICIPANT_NUMBERS, number_map, "test")

        # 你要求:train/test 不允许为空; val 允许为空
        if len(train_ids) == 0:
            raise RuntimeError("train 数据集 participant 数量不能为 0:请设置 TRAIN_PARTICIPANT_NUMBERS.")
        if len(test_ids) == 0:
            raise RuntimeError("test 数据集 participant 数量不能为 0:请设置 TEST_PARTICIPANT_NUMBERS.")

        # 你要求:同一 participant 不能同时属于多个 set
        s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
        overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
        if overlap:
            raise RuntimeError(
                "同一个 participant 不能同时出现在多个数据集中(会造成数据泄漏).\n"
                f"overlap={sorted(overlap)}"
            )

        split_method = (
            "ParticipantIdSplit("  # 为了论文复现,把参与者划分写入配置
            f"train={sorted(train_ids, key=natural_sort_key)}, "
            f"val={sorted(val_ids, key=natural_sort_key)}, "
            f"test={sorted(test_ids, key=natural_sort_key)}, "
            f"total={participant_total}, ref={ref_posture})"
        )
        split_debug: Dict[str, object] = {
            "mode": "participant",
            "reference_posture": ref_posture,
            "participant_total": participant_total,
            "participant_number_map": {str(k): v for k, v in sorted(number_map.items())},
            "train_participants": sorted(train_ids, key=natural_sort_key),
            "val_participants": sorted(val_ids, key=natural_sort_key),
            "test_participants": sorted(test_ids, key=natural_sort_key),
            "user_numbers": {
                "train": list(TRAIN_PARTICIPANT_NUMBERS),
                "validation": list(VALIDATION_PARTICIPANT_NUMBERS),
                "test": list(TEST_PARTICIPANT_NUMBERS),
            },
            "seed": int(seed),
        }
    else:
        # participant 数量过少,按你要求退化为样本级 0.7/0.15/0.15
        split_method = f"SampleShuffleSplit(train=0.7,val=0.15,test=0.15, participant_total={participant_total})"
        split_debug = {
            "mode": "sample",
            "reference_posture": ref_posture,
            "participant_total": participant_total,
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "seed": int(seed),
        }
        train_ids, val_ids, test_ids = [], [], []  # for type checker only

    # ----------------------------
    # 2) 逐 posture 读取并切分
    # ----------------------------
    X_train_list: List[np.ndarray] = []
    y_train_list: List[str] = []
    g_train_list: List[str] = []

    X_val_list: List[np.ndarray] = []
    y_val_list: List[str] = []
    g_val_list: List[str] = []

    X_test_list: List[np.ndarray] = []
    y_test_list: List[str] = []
    g_test_list: List[str] = []

    def _load_files_as_matrix(selected_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """读取一组 CSV 文件,合并为 (X, groups)."""
        posture_rows: List[np.ndarray] = []
        posture_groups: List[str] = []

        for f in selected_files:
            Xf = load_one_excel(f, feat_cfg)
            if Xf.size == 0:
                continue

            # 每个文件固定抽样 rows_per_file 行(可选)
            if rows_per_file is not None and len(Xf) > rows_per_file:
                idx = rng.choice(len(Xf), size=rows_per_file, replace=False)
                Xf = Xf[idx]

            posture_rows.append(Xf)
            posture_groups.append(extract_participant_group_id(f))

        if not posture_rows:
            return np.zeros((0, D), dtype=np.float32), np.array([], dtype=object)

        Xp = np.vstack(posture_rows)  # (M, D)

        gp = np.concatenate(
            [np.full(len(Xf), posture_groups[i], dtype=object) for i, Xf in enumerate(posture_rows)]
        )

        # posture 级别抽样(可选)
        if rows_per_posture is not None and len(Xp) > rows_per_posture:
            idx = rng.choice(len(Xp), size=rows_per_posture, replace=False)
            Xp = Xp[idx]
            gp = gp[idx]

        # 在读取完所有 participant 后随机打乱样本顺序
        if len(Xp) > 1:
            perm2 = rng.permutation(len(Xp))
            Xp = Xp[perm2]
            gp = gp[perm2]

        return Xp.astype(np.float32), gp.astype(object)

    for posture in postures:
        files = find_posture_sources_folder_only(data_root, posture)
        if not files:
            raise FileNotFoundError(
                f"未找到坐姿 {posture} 的数据文件.\n"
                f"请检查:{data_root / posture} 下是否包含 participant_*.csv"
            )

        # 兼容保留参数:max_files_per_posture
        # - participant 指定切分模式下(train/val/test 的 participant 由全局变量指定),因此这里不再额外截断
        # - 样本切分模式下(人数<3),仍然沿用旧逻辑:在 posture 内随机抽取 participant 文件
        if (not use_participant_split) and (max_files_per_posture is not None) and (len(files) > max_files_per_posture):
            pick = rng.permutation(len(files))[:max_files_per_posture]
            files = [files[i] for i in pick]

        # posture 内建立 group_id -> 文件 的映射
        file_map: Dict[str, Path] = {}
        for f in files:
            gid = extract_participant_group_id(f)
            if gid in file_map:
                raise RuntimeError(
                    f"posture={posture} 下出现重复 participant id:{gid}.\n"
                    f"请检查文件命名是否冲突."
                )
            file_map[gid] = f

        if use_participant_split:
            # 按 participant 指定切分(train_ids/val_ids/test_ids 由全局变量指定)
            def _ids_to_files(ids: List[str], set_name: str) -> List[Path]:
                missing = [pid for pid in ids if pid not in file_map]
                if missing:
                    raise RuntimeError(
                        f"posture={posture} 缺少部分 participant 文件,导致无法构建 {set_name} set.\n"
                        f"missing={missing}\n"
                        "请确认各 posture 的 participant 数量与命名一致."
                    )
                return [file_map[pid] for pid in ids]

            Xp_tr, gp_tr = _load_files_as_matrix(_ids_to_files(train_ids, "train"))

            if len(val_ids) > 0:
                Xp_va, gp_va = _load_files_as_matrix(_ids_to_files(val_ids, "validation"))
            else:
                Xp_va, gp_va = np.zeros((0, D), dtype=np.float32), np.array([], dtype=object)

            Xp_te, gp_te = _load_files_as_matrix(_ids_to_files(test_ids, "test"))

            if Xp_tr.shape[0] > 0:
                X_train_list.append(Xp_tr)
                y_train_list.extend([posture] * len(Xp_tr))
                g_train_list.extend(gp_tr.tolist())

            if Xp_va.shape[0] > 0:
                X_val_list.append(Xp_va)
                y_val_list.extend([posture] * len(Xp_va))
                g_val_list.extend(gp_va.tolist())

            if Xp_te.shape[0] > 0:
                X_test_list.append(Xp_te)
                y_test_list.extend([posture] * len(Xp_te))
                g_test_list.extend(gp_te.tolist())

        else:
            # participant 数量 <3:样本级随机切分 0.7/0.15/0.15
            Xp_all, gp_all = _load_files_as_matrix(files)

            if Xp_all.shape[0] == 0:
                raise RuntimeError(f"坐姿 {posture} 的所有文件读取后为空,请检查数据.")

            n = int(len(Xp_all))
            n_tr, n_va, n_te = _split_counts_by_ratio(n, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

            Xp_tr = Xp_all[:n_tr]
            gp_tr = gp_all[:n_tr]

            Xp_va = Xp_all[n_tr:n_tr + n_va]
            gp_va = gp_all[n_tr:n_tr + n_va]

            Xp_te = Xp_all[n_tr + n_va:n_tr + n_va + n_te]
            gp_te = gp_all[n_tr + n_va:n_tr + n_va + n_te]

            if Xp_tr.shape[0] > 0:
                X_train_list.append(Xp_tr)
                y_train_list.extend([posture] * len(Xp_tr))
                g_train_list.extend(gp_tr.tolist())

            if Xp_va.shape[0] > 0:
                X_val_list.append(Xp_va)
                y_val_list.extend([posture] * len(Xp_va))
                g_val_list.extend(gp_va.tolist())

            if Xp_te.shape[0] > 0:
                X_test_list.append(Xp_te)
                y_test_list.extend([posture] * len(Xp_te))
                g_test_list.extend(gp_te.tolist())

    def _finalize(
        X_list: List[np.ndarray],
        y_list: List[str],
        g_list: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X_list:
            X_all = np.vstack(X_list).astype(np.float32)
        else:
            X_all = np.zeros((0, D), dtype=np.float32)
        y_all = np.array(y_list, dtype=object)
        g_all = np.array(g_list, dtype=object)
        return X_all, y_all, g_all

    X_train, y_train, g_train = _finalize(X_train_list, y_train_list, g_train_list)
    X_val, y_val, g_val = _finalize(X_val_list, y_val_list, g_val_list)
    X_test, y_test, g_test = _finalize(X_test_list, y_test_list, g_test_list)

    # 兜底检查:train/test 不允许为空(否则无法训练/评估)
    if X_train.shape[0] == 0:
        raise RuntimeError("训练集为空:无法训练模型.请检查数据或切分参数.")
    if X_test.shape[0] == 0:
        raise RuntimeError("测试集为空:无法评估模型.请检查数据或切分参数.")

    return (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test), split_method, split_debug

class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    在训练/推理阶段对指定特征乘权重。
    注意：必须放在 RobustScaler 之后，否则会被 scaler 抵消。
    """
    def __init__(self, indices: Sequence[int], weights: Sequence[float]):
        self.indices = list(indices) if indices is not None else []
        self.weights = [float(w) for w in (weights if weights is not None else [])]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            return X
        if not self.indices:
            return X
        if len(self.indices) != len(self.weights):
            raise ValueError("FeatureWeighter: indices 与 weights 长度不一致")
        Xw = X.copy()
        for idx, w in zip(self.indices, self.weights):
            ii = int(idx)
            if 0 <= ii < Xw.shape[1]:
                Xw[:, ii] *= float(w)
        return Xw

# ============================================================
# 5) 训练配置:KNN (支持 group-aware CV 超参搜索)
# ============================================================

@dataclass
class ModelConfig:
    """KNN 的训练配置(会写入 JSON 以便复现)."""
    n_neighbors: int = KNN_N_NEIGHBORS
    weights: str = KNN_WEIGHTS
    metric: str = KNN_METRIC
    p: int = KNN_P
    algorithm: str = KNN_ALGORITHM
    leaf_size: int = KNN_LEAF_SIZE
    n_jobs: int = KNN_N_JOBS
    max_export_samples: int = KNN_MAX_EXPORT_SAMPLES  # 0=不限制


def train_and_evaluate(
    X_train: np.ndarray,          # (Ntr,D)
    y_train: np.ndarray,          # (Ntr,)
    groups_train: np.ndarray,     # (Ntr,)
    X_val: np.ndarray,            # (Nv,D)  可为空
    y_val: np.ndarray,            # (Nv,)
    groups_val: np.ndarray,       # (Nv,)
    X_test: np.ndarray,           # (Nte,D)
    y_test: np.ndarray,           # (Nte,)
    groups_test: np.ndarray,      # (Nte,)
    feature_names: Sequence[str],    # (D,) 训练/导出使用的特征名顺序(用于定位 FSR 维度)
    model_cfg: ModelConfig,
    seed: int,
    split_method: str,
    split_debug: Dict[str, object],
) -> Tuple[Pipeline, Dict[str, object], Dict[str, object]]:
    """
    训练并评估 KNeighborsClassifier,并返回 (pipeline, metrics, extras).

    保持与现有脚本一致的外部功能:
    - FSR 增强仍仅在训练阶段生效
    - RobustScaler 仍保留
    - 超参搜索仍使用 group-aware CV (GroupKFold / LeaveOneGroupOut, groups=participant)
    """

    # ========== 训练增强: FSR Sensor Masking / Domain Randomization ==========
    feature_names = list(feature_names) if feature_names is not None else []
    fsr_indices: List[int] = [feature_names.index(n) for n in FSR_FEATURE_NAMES if n in feature_names]

    use_fsr_augmentation = bool(fsr_indices) and (
        (bool(ENABLE_FSR_DOMAIN_RANDOMIZATION))
        or (bool(ENABLE_FSR_SENSOR_MASKING))
    )

    augmentation_cfg = {
        'enabled': bool(use_fsr_augmentation),
        'fsr_included': bool(fsr_indices),
        'fsr_indices': [int(i) for i in fsr_indices],
        'sensor_masking': {
            'enabled': bool(ENABLE_FSR_SENSOR_MASKING),
            'mask_prob': float(FSR_SENSOR_MASK_PROB),
        },
        'domain_randomization': {
            'enabled': bool(ENABLE_FSR_DOMAIN_RANDOMIZATION),
            'scale_range': tuple(float(x) for x in FSR_DR_SCALE_RANGE) if FSR_DR_SCALE_RANGE is not None else None,
            'noise_std': float(FSR_DR_NOISE_STD),
            'bias_std': float(FSR_DR_BIAS_STD),
        },
        'random_state': int(seed),
    }

    def _norm_metric(v: str) -> str:
        m = str(v).strip().lower()
        if m in ("cityblock", "l1"):
            return "manhattan"
        if m in ("l2",):
            return "euclidean"
        return m

    # 初始化(用户/默认)超参
    n_neighbors_init = max(1, int(getattr(model_cfg, 'n_neighbors', KNN_N_NEIGHBORS)))
    weights_init = str(getattr(model_cfg, 'weights', KNN_WEIGHTS)).strip().lower()
    metric_init = _norm_metric(str(getattr(model_cfg, 'metric', KNN_METRIC)))
    p_init = max(1, int(getattr(model_cfg, 'p', KNN_P)))
    algorithm_init = str(getattr(model_cfg, 'algorithm', KNN_ALGORITHM)).strip().lower()
    leaf_size_init = max(1, int(getattr(model_cfg, 'leaf_size', KNN_LEAF_SIZE)))
    n_jobs = int(getattr(model_cfg, 'n_jobs', KNN_N_JOBS))
    max_export_samples = int(getattr(model_cfg, 'max_export_samples', KNN_MAX_EXPORT_SAMPLES) or 0)

    if weights_init not in ("uniform", "distance"):
        raise ValueError(f"KNN weights 只能是 'uniform'/'distance', 但收到: {weights_init!r}")
    if metric_init not in ("minkowski", "euclidean", "manhattan"):
        raise ValueError(f"KNN metric 只能是 'minkowski'/'euclidean'/'manhattan', 但收到: {metric_init!r}")
    if algorithm_init not in ("auto", "ball_tree", "kd_tree", "brute"):
        raise ValueError(f"KNN algorithm 只能是 'auto'/'ball_tree'/'kd_tree'/'brute', 但收到: {algorithm_init!r}")

    # group 统计(用于 debug/日志)
    classes_tr = np.unique(y_train)
    group_cnt_per_class_train = {c: len(np.unique(groups_train[y_train == c])) for c in classes_tr} if len(classes_tr) else {}
    total_groups_train = int(len(np.unique(groups_train))) if groups_train is not None else 0
    g_all_list: List[np.ndarray] = []
    if groups_train is not None and len(groups_train) > 0:
        g_all_list.append(groups_train)
    if groups_val is not None and len(groups_val) > 0:
        g_all_list.append(groups_val)
    if groups_test is not None and len(groups_test) > 0:
        g_all_list.append(groups_test)
    if g_all_list:
        total_groups_all = int(len(np.unique(np.concatenate(g_all_list).astype(object))))
    else:
        total_groups_all = 0

    # 最终训练集 = train(+val)
    if X_val is not None and X_val.shape[0] > 0:
        X_fit = np.vstack([X_train, X_val]).astype(np.float32)
        y_fit = np.concatenate([y_train, y_val]).astype(object)
        g_fit = np.concatenate([groups_train, groups_val]).astype(object)
    else:
        X_fit = X_train.astype(np.float32)
        y_fit = y_train.astype(object)
        g_fit = groups_train.astype(object)

    cv_enabled = bool(ENABLE_CROSS_VALIDATION)

    # ========== 交叉验证器: GroupKFold / LeaveOneGroupOut ==========
    def _make_group_cv(groups: np.ndarray) -> Tuple[Optional[object], str]:
        uniq = np.unique(groups.astype(object))
        n_g = int(len(uniq))
        if n_g < 2:
            return None, 'no_group_cv(n_groups<2)'

        strat = str(CROSS_VALIDATION_STRATEGY).strip().lower()
        if strat not in ("auto", "logo", "groupkfold"):
            raise ValueError(
                "CROSS_VALIDATION_STRATEGY 只能是 'auto'/'logo'/'groupkfold',"
                f" 但收到: {CROSS_VALIDATION_STRATEGY!r}"
            )

        if strat == "logo":
            return LeaveOneGroupOut(), f'LeaveOneGroupOut(n_splits={n_g})'

        if strat == "groupkfold":
            k = int(min(max(2, int(CV_GROUPKFOLD_N_SPLITS)), n_g))
            return GroupKFold(n_splits=k), f'GroupKFold(n_splits={k})'

        if n_g <= int(CV_LOGO_MAX_GROUPS):
            return LeaveOneGroupOut(), f'LeaveOneGroupOut(n_splits={n_g})'
        k = int(min(max(2, int(CV_GROUPKFOLD_N_SPLITS)), n_g))
        return GroupKFold(n_splits=k), f'GroupKFold(n_splits={k})'

    if cv_enabled:
        cv, cv_desc = _make_group_cv(g_fit)
    else:
        cv, cv_desc = None, 'disabled'

    def _make_fallback_holdout(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
        tr_idx, va_idx = next(splitter.split(X, y))
        return tr_idx.astype(int), va_idx.astype(int)

    # helper: build pipelines
    def _pipeline_steps_with_optional_aug(clf_obj) -> List[Tuple[str, object]]:
        steps: List[Tuple[str, object]] = []
        if use_fsr_augmentation:
            steps.append((
                'fsr_aug',
                FsrAugmenter(
                    enable_sensor_masking=bool(ENABLE_FSR_SENSOR_MASKING),
                    mask_prob=float(FSR_SENSOR_MASK_PROB),
                    enable_domain_randomization=bool(ENABLE_FSR_DOMAIN_RANDOMIZATION),
                    scale_range=FSR_DR_SCALE_RANGE,
                    noise_std=float(FSR_DR_NOISE_STD),
                    bias_std=float(FSR_DR_BIAS_STD),
                    fsr_indices=fsr_indices,
                    random_state=int(seed),
                ),
            ))

        steps.append(('scaler', RobustScaler()))

        weight_indices: List[int] = []
        weight_values: List[float] = []
        if bool(ENABLE_FEATURE_WEIGHTING):
            if 'tof_gated' in feature_names:
                weight_indices.append(int(feature_names.index('tof_gated')))
                weight_values.append(float(TOF_GATED_WEIGHT))
        if weight_indices:
            steps.append(('feat_weight', FeatureWeighter(weight_indices, weight_values)))

        steps.append(('clf', clf_obj))
        return steps

    def _resolve_k(k_req: int, n_train: int) -> int:
        return max(1, min(int(k_req), max(1, int(n_train))))

    def _pipe_knn(
        n_neighbors: int,
        weights: str,
        metric: str,
        p: int,
        algorithm: str,
        leaf_size: int,
        n_train: int,
    ) -> Pipeline:
        k_eff = _resolve_k(n_neighbors, n_train)
        metric_use = _norm_metric(metric)
        p_use = int(p) if metric_use == "minkowski" else 2
        clf = KNeighborsClassifier(
            n_neighbors=int(k_eff),
            weights=str(weights),
            algorithm=str(algorithm),
            leaf_size=int(leaf_size),
            p=int(max(1, p_use)),
            metric=str(metric_use),
            n_jobs=int(n_jobs),
        )
        return Pipeline(steps=_pipeline_steps_with_optional_aug(clf))

    def _cv_score(
        n_neighbors: int,
        weights: str,
        metric: str,
        p: int,
        algorithm: str,
        leaf_size: int,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[float, float, int]:
        """返回 (mean_acc, std_acc, n_folds)."""
        scores: List[float] = []
        if cv is not None:
            splits = list(cv.split(X, y, groups=groups))
        else:
            tr_idx, va_idx = _make_fallback_holdout(X, y)
            splits = [(tr_idx, va_idx)]

        for tr_idx, va_idx in splits:
            pipe = _pipe_knn(n_neighbors, weights, metric, p, algorithm, leaf_size, n_train=len(tr_idx))
            pipe.fit(X[tr_idx], y[tr_idx])
            pred = pipe.predict(X[va_idx])
            scores.append(float(accuracy_score(y[va_idx], pred)))
        return float(np.mean(scores)) if scores else 0.0, float(np.std(scores)) if scores else 0.0, int(len(scores))

    # ========== 超参搜索(KNN) ==========
    knn_search_debug: Dict[str, object] = {
        'method': 'group_cv' if cv is not None else ('fallback_holdout' if cv_enabled else 'disabled'),
        'cv': cv_desc,
        'search_max_trials': int(KNN_SEARCH_MAX_TRIALS),
        'max_export_samples': int(max_export_samples),
    }

    def _uniq(seq):
        out = []
        seen = set()
        for x in seq:
            key = str(x)
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    k_list = [int(x) for x in _uniq([n_neighbors_init] + list(KNN_SEARCH_N_NEIGHBORS_LIST)) if int(x) >= 1]
    w_list = [str(x).strip().lower() for x in _uniq([weights_init] + list(KNN_SEARCH_WEIGHTS_LIST)) if str(x).strip().lower() in ("uniform", "distance")]
    m_list = [_norm_metric(str(x)) for x in _uniq([metric_init] + list(KNN_SEARCH_METRIC_LIST)) if _norm_metric(str(x)) in ("minkowski", "euclidean", "manhattan")]
    p_list = [int(x) for x in _uniq([p_init] + list(KNN_SEARCH_P_LIST)) if int(x) >= 1]
    a_list = [str(x).strip().lower() for x in _uniq([algorithm_init] + list(KNN_SEARCH_ALGORITHM_LIST)) if str(x).strip().lower() in ("auto", "ball_tree", "kd_tree", "brute")]
    leaf_list = [int(x) for x in _uniq([leaf_size_init] + list(KNN_SEARCH_LEAF_SIZE_LIST)) if int(x) >= 1]

    knn_search_debug['candidates'] = {
        'n_neighbors': [int(x) for x in k_list],
        'weights': [str(x) for x in w_list],
        'metric': [str(x) for x in m_list],
        'p': [int(x) for x in p_list],
        'algorithm': [str(x) for x in a_list],
        'leaf_size': [int(x) for x in leaf_list],
    }

    grid: List[Tuple[int, str, str, int, str, int]] = []
    for k in k_list:
        for w in w_list:
            for m in m_list:
                for p in p_list:
                    if m != "minkowski" and int(p) != 2:
                        continue
                    for a in a_list:
                        for leaf in leaf_list:
                            grid.append((int(k), str(w), str(m), int(p), str(a), int(leaf)))

    knn_search_debug['grid_size'] = int(len(grid))

    rng = np.random.default_rng(int(seed))
    if cv_enabled and (len(grid) > int(KNN_SEARCH_MAX_TRIALS) > 0):
        idx = rng.choice(len(grid), size=int(KNN_SEARCH_MAX_TRIALS), replace=False)
        grid = [grid[i] for i in idx]
        knn_search_debug['grid_sampled'] = True
        knn_search_debug['grid_sampled_size'] = int(len(grid))
    else:
        knn_search_debug['grid_sampled'] = False
        knn_search_debug['grid_sampled_size'] = int(len(grid))

    selected = {
        'n_neighbors': int(n_neighbors_init),
        'weights': str(weights_init),
        'metric': str(metric_init),
        'p': int(p_init),
        'algorithm': str(algorithm_init),
        'leaf_size': int(leaf_size_init),
    }

    tried: List[Dict[str, object]] = []

    if cv_enabled:
        # best tuple: (mean_acc, std_acc, n_neighbors_eff, params)
        best: Optional[Tuple[float, float, int, Dict[str, object]]] = None

        for (k, w, m, p, a, leaf) in grid:
            try:
                mean_acc, std_acc, n_folds = _cv_score(k, w, m, p, a, leaf, X_fit, y_fit, g_fit)
                k_eff_fit = _resolve_k(k, X_fit.shape[0])
                rec = {
                    'n_neighbors': int(k),
                    'n_neighbors_effective_fit': int(k_eff_fit),
                    'weights': str(w),
                    'metric': str(m),
                    'p': int(p),
                    'algorithm': str(a),
                    'leaf_size': int(leaf),
                    'cv_mean_acc': float(mean_acc),
                    'cv_std': float(std_acc),
                    'n_folds': int(n_folds),
                }
                tried.append(rec)

                params = {
                    'n_neighbors': int(k),
                    'weights': str(w),
                    'metric': str(m),
                    'p': int(p),
                    'algorithm': str(a),
                    'leaf_size': int(leaf),
                }

                if best is None:
                    best = (float(mean_acc), float(std_acc), int(k_eff_fit), params)
                else:
                    if (mean_acc > best[0]) or \
                       (mean_acc == best[0] and std_acc < best[1]) or \
                       (mean_acc == best[0] and std_acc == best[1] and int(k_eff_fit) < best[2]):
                        best = (float(mean_acc), float(std_acc), int(k_eff_fit), params)
            except Exception as e:
                tried.append({
                    'n_neighbors': int(k),
                    'weights': str(w),
                    'metric': str(m),
                    'p': int(p),
                    'algorithm': str(a),
                    'leaf_size': int(leaf),
                    'error': str(e),
                })

        knn_search_debug['knn_search'] = tried

        if best is not None:
            selected = best[3]
            knn_search_debug['selected'] = {
                **selected,
                'cv_mean_acc': float(best[0]),
                'cv_std': float(best[1]),
                'n_neighbors_effective_fit': int(best[2]),
            }
        else:
            knn_search_debug['selected'] = {
                'fallback': 'init_params(all_candidates_failed)',
                **selected,
            }
    else:
        knn_search_debug['knn_search'] = 'disabled'
        knn_search_debug['selected'] = selected

    # ========== 构建最终模型并训练 ==========
    pipeline = _pipe_knn(
        n_neighbors=int(selected['n_neighbors']),
        weights=str(selected['weights']),
        metric=str(selected['metric']),
        p=int(selected['p']),
        algorithm=str(selected['algorithm']),
        leaf_size=int(selected['leaf_size']),
        n_train=int(X_fit.shape[0]),
    )
    pipeline.fit(X_fit, y_fit)

    # 导出样本量兜底检查
    clf_final: KNeighborsClassifier = pipeline.named_steps['clf']
    X_fit_mem = np.asarray(getattr(clf_final, '_fit_X', np.zeros((0, X_fit.shape[1]))), dtype=np.float32)
    n_train_samples = int(X_fit_mem.shape[0])
    if (max_export_samples > 0) and (n_train_samples > int(max_export_samples)):
        raise RuntimeError(
            "KNN 训练样本数超过 max_export_samples 上限,无法导出/部署.\n"
            f"n_train_samples={n_train_samples}, max_export_samples={max_export_samples}\n"
            "请减少训练样本(例如 rows_per_posture/rows_per_file),或调大 KNN_MAX_EXPORT_SAMPLES."
        )

    # ========== 测试集评估 ==========
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report_text = classification_report(y_test, y_pred, digits=4, output_dict=False)

    classes_order = list(getattr(clf_final, 'classes_', sorted(list(np.unique(y_fit)))))
    cm = confusion_matrix(y_test, y_pred, labels=classes_order)
    test_counts = [int(np.sum(y_test == lbl)) for lbl in classes_order]

    metrics = {
        'model': 'knn',
        'accuracy': acc,
        'classification_report_text': report_text,
        'labels': classes_order,
        'confusion_matrix': cm,
        'test_counts': test_counts,

        # dataset split (external)
        'split_method': split_method,
        'split_debug': split_debug,

        # training augmentation
        'augmentation': augmentation_cfg,

        # group statistics
        'group_cnt_per_class_train': group_cnt_per_class_train,
        'unique_groups_train': int(total_groups_train),
        'unique_groups_all': int(total_groups_all),

        # cross validation
        'cv_enabled': bool(cv_enabled),
        'cv_strategy': str(CROSS_VALIDATION_STRATEGY),
        'cv_desc': cv_desc,

        # final hyper-params used
        'n_neighbors_requested': int(selected['n_neighbors']),
        'n_neighbors_used': int(getattr(clf_final, 'n_neighbors', selected['n_neighbors'])),
        'weights_used': str(selected['weights']),
        'metric_used': str(_norm_metric(selected['metric'])),
        'p_used': int(selected['p']) if str(_norm_metric(selected['metric'])) == "minkowski" else 2,
        'algorithm_used': str(selected['algorithm']),
        'leaf_size_used': int(selected['leaf_size']),
        'n_jobs': int(n_jobs),
        'max_export_samples': int(max_export_samples),
        'n_train_samples': int(n_train_samples),

        # search debug
        'knn_search_debug': knn_search_debug,
    }

    extras = {
        'X_train_shape': X_train.shape,
        'X_val_shape': X_val.shape if X_val is not None else (0, X_train.shape[1]),
        'X_test_shape': X_test.shape,
        'X_fit_shape': X_fit.shape,
    }
    return pipeline, metrics, extras
def _fmt_f(x: float) -> str:
    """格式化为 C float 常量(带 f 后缀)."""
    return f"{float(x):.9g}f"


def _append_c_standardize_block(
    lines: List[str],
    feature_names: Sequence[str],
):
    lines.append("  float x[AML_FEATURES];")
    lines.append("  // Copy input")
    lines.append("  for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("    x[j] = x_in[j];")
    lines.append("  }")

    lines.append("")
    lines.append("  // Robust scale (RobustScaler)")
    lines.append("  for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("    x[j] = (x[j] - AML_SCALER_MEAN[j]) / AML_SCALER_SCALE[j];")
    lines.append("  }")

    # ---------- NEW: feature weighting (must match training pipeline) ----------
    try:
        idx = list(feature_names).index("tof_gated")
    except ValueError:
        idx = None

    if bool(ENABLE_FEATURE_WEIGHTING) and (idx is not None):
        lines.append("")
        lines.append("  // Feature weighting (must match training)")
        lines.append(f"  // tof_gated (x[{idx}]) *= {float(TOF_GATED_WEIGHT):.6g}")
        lines.append(f"  x[{idx}] *= {_fmt_f(float(TOF_GATED_WEIGHT))};")




def export_knn_to_c_header(
    pipeline: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    out_path: Path,
):
    """
    导出 KNeighborsClassifier 到 C 头文件.

    约束/约定:
    - MCU 端输入必须是最终特征(D 维),顺序与 feature_names 完全一致
    - MCU 端做: RobustScaler + (可选)feature weighting + KNN brute-force 推理
    - 当前导出支持 metric: euclidean / manhattan / minkowski
      (minkowski 的 p!=1/2 也支持)
    """
    scaler: RobustScaler = pipeline.named_steps["scaler"]
    clf: KNeighborsClassifier = pipeline.named_steps["clf"]

    if list(getattr(clf, "classes_", [])) != class_names:
        class_names = list(getattr(clf, "classes_", class_names))

    n_feat = int(len(scaler.center_))
    n_cls = int(len(class_names))

    mean = scaler.center_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    fit_X = np.asarray(getattr(clf, "_fit_X", None), dtype=np.float32)
    if fit_X is None or fit_X.ndim != 2 or fit_X.shape[0] == 0:
        raise RuntimeError("导出失败: KNN 模型内部训练样本为空(clf._fit_X 不可用).")
    if int(fit_X.shape[1]) != int(n_feat):
        raise RuntimeError(
            f"导出失败: clf._fit_X 维度与 scaler 不一致: "
            f"{fit_X.shape[1]} vs {n_feat}."
        )

    fit_y_raw = np.asarray(getattr(clf, "_y", None)).reshape(-1)
    if fit_y_raw.size != fit_X.shape[0]:
        raise RuntimeError(
            f"导出失败: KNN 内部标签数与样本数不一致: "
            f"len(_y)={fit_y_raw.size}, n_samples={fit_X.shape[0]}."
        )

    if np.issubdtype(fit_y_raw.dtype, np.integer):
        fit_y = fit_y_raw.astype(np.int32)
    else:
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        fit_y = np.array([class_to_idx.get(v, 0) for v in fit_y_raw.tolist()], dtype=np.int32)

    if np.any(fit_y < 0) or np.any(fit_y >= n_cls):
        raise RuntimeError("导出失败: KNN 内部标签索引超出类别范围.")

    n_train = int(fit_X.shape[0])
    k = int(max(1, min(int(getattr(clf, "n_neighbors", 1)), n_train)))

    weights = str(getattr(clf, "weights", "uniform")).strip().lower()
    if weights not in ("uniform", "distance"):
        raise RuntimeError(f"导出失败: 当前仅支持 weights=uniform/distance,但收到 {weights!r}.")
    weight_distance = (weights == "distance")

    metric = str(getattr(clf, "metric", "minkowski")).strip().lower()
    p = int(getattr(clf, "p", 2) or 2)
    if metric in ("cityblock", "l1"):
        metric = "manhattan"
    elif metric in ("l2",):
        metric = "euclidean"

    metric_kind = 2  # 0=euclidean, 1=manhattan, 2=minkowski
    if metric == "euclidean":
        metric_kind = 0
        p = 2
    elif metric == "manhattan":
        metric_kind = 1
        p = 1
    elif metric == "minkowski":
        metric_kind = 2
        p = max(1, int(p))
    else:
        raise RuntimeError(
            f"导出失败: 当前仅支持 metric=euclidean/manhattan/minkowski,但收到 {metric!r}."
        )

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    lines: List[str] = []
    lines.append(f"// Auto-generated by AML_KNN.py on {now}")
    lines.append(f"// Input: {n_feat} features (no feature engineering on MCU)")
    lines.append("// Model: KNeighborsClassifier (brute-force inference on MCU)")
    lines.append(f"// Train samples: {n_train}, classes: {n_cls}")
    lines.append(f"// K: {k}, weights: {weights}, metric: {metric}, p: {p}")
    lines.append("// Feature order expected by aml_predict_class():")
    for i, name in enumerate(feature_names):
        lines.append(f"//   x[{i:2d}] = {name}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <float.h>")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("extern \"C\" {")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#define AML_FEATURES {n_feat}")
    lines.append(f"#define AML_NUM_CLASSES {n_cls}")
    lines.append(f"#define AML_KNN_TRAIN_N {n_train}")
    lines.append(f"#define AML_KNN_K {k}")
    lines.append(f"#define AML_KNN_WEIGHT_DISTANCE {1 if weight_distance else 0}")
    lines.append(f"#define AML_KNN_METRIC_KIND {metric_kind}")
    lines.append(f"#define AML_KNN_P {_fmt_f(float(p))}")
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
    lines.append("static const float AML_KNN_X[AML_KNN_TRAIN_N][AML_FEATURES] = {")
    for i in range(n_train):
        row = ", ".join(_fmt_f(v) for v in fit_X[i])
        lines.append(f"  {{ {row} }},")
    lines.append("};")
    lines.append("static const int16_t AML_KNN_Y[AML_KNN_TRAIN_N] = {")
    lines.append("  " + ", ".join(str(int(v)) for v in fit_y.tolist()) + ",")
    lines.append("};")

    lines.append("\n\n// =============================")
    lines.append("// 推理:标准化 + KNN 暴力检索 + 投票")
    lines.append("// =============================")
    lines.append("\nstatic inline int aml_predict_class(const float x_in[AML_FEATURES]) {")
    _append_c_standardize_block(lines, feature_names)
    lines.append("")
    lines.append("  float best_dist[AML_KNN_K];")
    lines.append("  int16_t best_cls[AML_KNN_K];")
    lines.append("  for (int t = 0; t < AML_KNN_K; ++t) {")
    lines.append("    best_dist[t] = FLT_MAX;")
    lines.append("    best_cls[t] = -1;")
    lines.append("  }")
    lines.append("")
    lines.append("  for (int i = 0; i < AML_KNN_TRAIN_N; ++i) {")
    lines.append("    float dist = 0.0f;")
    lines.append("    if (AML_KNN_METRIC_KIND == 0) {")
    lines.append("      float acc = 0.0f;")
    lines.append("      for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("        float d = x[j] - AML_KNN_X[i][j];")
    lines.append("        acc += d * d;")
    lines.append("      }")
    lines.append("      dist = sqrtf(acc);")
    lines.append("    } else if (AML_KNN_METRIC_KIND == 1) {")
    lines.append("      float acc = 0.0f;")
    lines.append("      for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("        float d = x[j] - AML_KNN_X[i][j];")
    lines.append("        acc += fabsf(d);")
    lines.append("      }")
    lines.append("      dist = acc;")
    lines.append("    } else {")
    lines.append("      float acc = 0.0f;")
    lines.append("      for (int j = 0; j < AML_FEATURES; ++j) {")
    lines.append("        float d = fabsf(x[j] - AML_KNN_X[i][j]);")
    lines.append("        acc += powf(d, AML_KNN_P);")
    lines.append("      }")
    lines.append("      dist = powf(acc, 1.0f / AML_KNN_P);")
    lines.append("    }")
    lines.append("")
    lines.append("    if (dist < best_dist[AML_KNN_K - 1]) {")
    lines.append("      int pos = AML_KNN_K - 1;")
    lines.append("      while (pos > 0 && dist < best_dist[pos - 1]) {")
    lines.append("        best_dist[pos] = best_dist[pos - 1];")
    lines.append("        best_cls[pos] = best_cls[pos - 1];")
    lines.append("        --pos;")
    lines.append("      }")
    lines.append("      best_dist[pos] = dist;")
    lines.append("      best_cls[pos] = AML_KNN_Y[i];")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  float class_score[AML_NUM_CLASSES];")
    lines.append("  for (int c = 0; c < AML_NUM_CLASSES; ++c) class_score[c] = 0.0f;")
    lines.append("")
    lines.append("  for (int t = 0; t < AML_KNN_K; ++t) {")
    lines.append("    int cls = (int)best_cls[t];")
    lines.append("    if (cls < 0 || cls >= AML_NUM_CLASSES) continue;")
    lines.append("    float w = 1.0f;")
    lines.append("    if (AML_KNN_WEIGHT_DISTANCE) {")
    lines.append("      float d = best_dist[t];")
    lines.append("      if (d < 1e-12f) d = 1e-12f;")
    lines.append("      w = 1.0f / d;")
    lines.append("    }")
    lines.append("    class_score[cls] += w;")
    lines.append("  }")
    lines.append("")
    lines.append("  int best = 0;")
    lines.append("  float best_v = class_score[0];")
    lines.append("  for (int c = 1; c < AML_NUM_CLASSES; ++c) {")
    lines.append("    if (class_score[c] > best_v) {")
    lines.append("      best_v = class_score[c];")
    lines.append("      best = c;")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return best;")
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
    """统一导出入口:根据分类器类型选择导出格式."""
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, KNeighborsClassifier):
        export_knn_to_c_header(pipeline, feature_names, class_names, out_path)
        return

    raise TypeError(f"不支持导出的分类器类型:{type(clf)}")
# ============================================================
# 7) 可视化:保存混淆矩阵图片
# ============================================================

def save_confusion_matrix_standard(
    cm: np.ndarray,
    labels: List[str],
    test_counts: Optional[List[int]],
    out_path: Path,
):
    """
    论文标准混淆矩阵绘图:
    - 只绘制 K×K 的混淆矩阵
    - 测试集样本数量以文字形式写在矩阵下方
    """
    # 为避免在“无显示环境”(如 SSH/CI)下因 matplotlib 后端导致报错,固定使用 Agg 后端
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



def confusion_matrix_to_text(cm: np.ndarray, labels: Sequence[str]) -> str:
    """
    将混淆矩阵格式化为终端友好的文本表格.

    行(True): 真实标签
    列(Pred): 预测标签
    """
    cm = np.asarray(cm, dtype=np.int64)
    labels = [str(x) for x in labels]
    k = len(labels)

    if cm.size == 0 or k == 0:
        return "<empty confusion matrix>"

    # 单元格宽度:兼顾数字宽度与标签宽度
    num_w = max(len(str(int(v))) for v in cm.flatten())
    lab_w = max(len(s) for s in labels)
    cell_w = max(5, num_w + 2, lab_w + 2)

    # 表头
    head = " " * (lab_w + 2) + "".join(f"{s:>{cell_w}}" for s in labels) + f"{'Σ':>{cell_w}}"
    lines = [head]

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    total = int(cm.sum())

    # 每行
    for i, s in enumerate(labels):
        row = "".join(f"{int(cm[i, j]):>{cell_w}d}" for j in range(k))
        lines.append(f"{s:<{lab_w}}  {row}{int(row_sums[i]):>{cell_w}d}")

    # 列和
    foot = f"{'Σ':<{lab_w}}  " + "".join(f"{int(col_sums[j]):>{cell_w}d}" for j in range(k)) + f"{total:>{cell_w}d}"
    lines.append(foot)
    return "\n".join(lines)



# ============================================================
# 8) main
# ============================================================

def main():
    # 禁止 data_root/output_dir 命令行覆盖(按你要求)
    for forbidden in FORBIDDEN_CLI_ARGS:
        if forbidden in sys.argv:
            raise RuntimeError(
                f"已禁用命令行参数 {forbidden}.\n"
                f"请在代码中修改 DATA_ROOT_PATH / OUTPUT_ROOT_PATH(位于 import 后的可调参数区域)."
            )

    parser = argparse.ArgumentParser(
        description="训练坐姿 KNN 并导出为 ESP32/Arduino 可用的 C 头文件(tof_gated/tof_use 由采集端提供)"
    )

    parser.add_argument(
        "--postures",
        type=str,
        default="",
        help="逗号分隔的 posture 列表;为空则自动扫描 data_root 下的子文件夹(全部参与训练)",
    )

    parser.add_argument("--max_files_per_posture", type=int, default=DEFAULT_MAX_FILES_PER_POSTURE)
    parser.add_argument("--rows_per_posture", type=int, default=DEFAULT_ROWS_PER_POSTURE)
    parser.add_argument("--rows_per_file", type=int, default=DEFAULT_ROWS_PER_FILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no_confirm", action="store_true", default=DEFAULT_NO_CONFIRM)

    # ---------- KNN 模型参数 ----------
    parser.add_argument("--n_neighbors", type=int, default=KNN_N_NEIGHBORS, help="(KNN) 邻居数 k")
    parser.add_argument("--weights", type=str, default=KNN_WEIGHTS, choices=["uniform", "distance"], help="(KNN) 投票权重")
    parser.add_argument("--metric", type=str, default=KNN_METRIC, choices=["minkowski", "euclidean", "manhattan"], help="(KNN) 距离度量")
    parser.add_argument("--p", type=int, default=KNN_P, help="(KNN) minkowski 的 p 参数(metric=minkowski 时生效)")
    parser.add_argument("--algorithm", type=str, default=KNN_ALGORITHM, choices=["auto", "ball_tree", "kd_tree", "brute"], help="(KNN) 邻居搜索算法")
    parser.add_argument("--leaf_size", type=int, default=KNN_LEAF_SIZE, help="(KNN) leaf_size")
    parser.add_argument("--n_jobs", type=int, default=KNN_N_JOBS, help="(KNN) 并行线程数; -1=全部核")
    parser.add_argument("--max_export_samples", type=int, default=KNN_MAX_EXPORT_SAMPLES, help="(KNN) 导出样本数上限; 0=不限制")

    args = parser.parse_args()

    data_root = Path(DATA_ROOT_PATH).expanduser().resolve()
    out_dir = Path(OUTPUT_ROOT_PATH).expanduser().resolve()
    if EXPORT_MODEL_FILES:
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

    # ----------------------------
    # 特征选择:
    # - 完全由 import 后的 SELECTED_FEATURE_KEYS 控制
    # - 本脚本不再提供/兼容命令行参数覆盖(避免训练/导出与采集端不一致)
    # ----------------------------
    feat_cfg = FeatureBuildConfig(feature_keys=list(SELECTED_FEATURE_KEYS))

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
    print(f"导出开关 EXPORT_MODEL_FILES: {EXPORT_MODEL_FILES} (True=写文件, False=不写文件)")
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
    print(f"SELECTED_FEATURE_KEYS: {feat_cfg.feature_keys}")
    print(f"输入维度 D: {len(model_feature_names)}")
    print("输入特征顺序:")
    for i, n in enumerate(model_feature_names):
        print(f"  x[{i:2d}] = {n}")

    print("\n================ 训练增强配置(FSR) ================")
    print(f"ENABLE_FSR_DOMAIN_RANDOMIZATION: {ENABLE_FSR_DOMAIN_RANDOMIZATION}")
    print(f"FSR_DR_SCALE_RANGE: {FSR_DR_SCALE_RANGE}")
    print(f"FSR_DR_NOISE_STD: {FSR_DR_NOISE_STD}")
    print(f"FSR_DR_BIAS_STD: {FSR_DR_BIAS_STD}")
    print(f"ENABLE_FSR_SENSOR_MASKING: {ENABLE_FSR_SENSOR_MASKING}")
    print(f"FSR_SENSOR_MASK_PROB: {FSR_SENSOR_MASK_PROB}")
    if "FSR" not in feat_cfg.feature_keys:
        print("注意: 当前 SELECTED_FEATURE_KEYS 不包含 FSR,上述增强将自动跳过.")

    if not args.no_confirm:
        if not ask_yes_no("\n确认以上数据无误,是否继续开始训练？", default=True):
            print("你选择了停止:脚本将退出(未进行训练/导出).")
            return

    print("\n================ 数据集切分配置 ================")
    print(f"TRAIN_PARTICIPANT_NUMBERS: {TRAIN_PARTICIPANT_NUMBERS}")
    print(f"VALIDATION_PARTICIPANT_NUMBERS: {VALIDATION_PARTICIPANT_NUMBERS}")
    print(f"TEST_PARTICIPANT_NUMBERS: {TEST_PARTICIPANT_NUMBERS}")
    print("说明: participant>=3 -> 按上面的 participant 指定划分; participant<3 -> 样本级随机划分(0.7/0.15/0.15,不变)")

    (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test), split_method, split_debug = build_dataset_splits(
        data_root=data_root,
        postures=selected_postures,
        max_files_per_posture=max_files,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=args.seed,
        feat_cfg=feat_cfg,
    )

    print("\n================ 交叉验证开关 ================")
    print(f"ENABLE_CROSS_VALIDATION: {ENABLE_CROSS_VALIDATION}")
    print(f"CROSS_VALIDATION_STRATEGY: {CROSS_VALIDATION_STRATEGY}")
    print(f"CV_GROUPKFOLD_N_SPLITS: {CV_GROUPKFOLD_N_SPLITS}")
    print(f"CV_LOGO_MAX_GROUPS: {CV_LOGO_MAX_GROUPS}")

    print("\n================ 数据集概览 ================")
    print(f"Split method: {split_method}")
    if split_debug:
        print("Split debug (简要):")
        print("  mode:", split_debug.get("mode"))
        print("  reference_posture:", split_debug.get("reference_posture"))
        print("  participant_total:", split_debug.get("participant_total"))
        if split_debug.get("mode") == "participant":
            print("  train_participants:", len(split_debug.get("train_participants", [])))
            print("  val_participants:", len(split_debug.get("val_participants", [])))
            print("  test_participants:", len(split_debug.get("test_participants", [])))
        else:
            print("  ratios:", split_debug.get("ratios"))

    print("\n[TRAIN]")
    print(f"X_train shape: {X_train.shape} (N,D)")
    uniq, cnt = np.unique(y_train, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  {u:20s}: {c}")
    print(f"Unique participants(groups): {len(np.unique(g_train))}")

    if X_val.shape[0] > 0:
        print("\n[VALIDATION]")
        print(f"X_val shape: {X_val.shape} (N,D)")
        uniq, cnt = np.unique(y_val, return_counts=True)
        for u, c in zip(uniq, cnt):
            print(f"  {u:20s}: {c}")
        print(f"Unique participants(groups): {len(np.unique(g_val))}")
    else:
        print("\n[VALIDATION] (empty)")

    print("\n[TEST]")
    print(f"X_test shape: {X_test.shape} (N,D)")
    uniq, cnt = np.unique(y_test, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  {u:20s}: {c}")
    print(f"Unique participants(groups): {len(np.unique(g_test))}")

    model_cfg = ModelConfig(
        n_neighbors=max(1, int(args.n_neighbors)),
        weights=str(args.weights).strip().lower(),
        metric=str(args.metric).strip().lower(),
        p=max(1, int(args.p)),
        algorithm=str(args.algorithm).strip().lower(),
        leaf_size=max(1, int(args.leaf_size)),
        n_jobs=int(args.n_jobs),
        max_export_samples=max(0, int(args.max_export_samples)),
    )

    pipeline, metrics, _ = train_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        groups_train=g_train,
        X_val=X_val,
        y_val=y_val,
        groups_val=g_val,
        X_test=X_test,
        y_test=y_test,
        groups_test=g_test,
        feature_names=model_feature_names,
        model_cfg=model_cfg,
        seed=args.seed,
        split_method=split_method,
        split_debug=split_debug,
    )

    print("\n================ 评估结果 ================")
    print(f"Model: {metrics['model']}")
    print(f"Split method: {metrics['split_method']}")
    if metrics.get("split_debug"):
        dbg = metrics["split_debug"]
        print("Split debug (简要):")
        print("  mode:", dbg.get("mode"))
        print("  reference_posture:", dbg.get("reference_posture"))
        print("  participant_total:", dbg.get("participant_total"))
        if dbg.get("mode") == "participant":
            print("  train_participants:", len(dbg.get("train_participants", [])))
            print("  val_participants:", len(dbg.get("val_participants", [])))
            print("  test_participants:", len(dbg.get("test_participants", [])))
        else:
            print("  ratios:", dbg.get("ratios"))

    print("\nKNN stats:")
    print(f"  train_samples: {metrics.get('n_train_samples')}")
    print(f"  max_export_samples: {metrics.get('max_export_samples')}")
    print("KNN params used:")
    print(
        "  "
        f"n_neighbors(req/used)={metrics.get('n_neighbors_requested')}/{metrics.get('n_neighbors_used')}, "
        f"weights={metrics.get('weights_used')}, "
        f"metric={metrics.get('metric_used')}, "
        f"p={metrics.get('p_used')}, "
        f"algorithm={metrics.get('algorithm_used')}, "
        f"leaf_size={metrics.get('leaf_size_used')}, "
        f"n_jobs={metrics.get('n_jobs')}"
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test error (1-accuracy): {1.0 - metrics['accuracy']:.4f}")
    print(metrics["classification_report_text"])

    print("\nConfusion matrix (rows=True, cols=Pred):")
    print(confusion_matrix_to_text(metrics["confusion_matrix"], metrics["labels"]))

    if EXPORT_MODEL_FILES:
        cm_path = out_dir / OUT_CM_NAME
        header_path = out_dir / OUT_HEADER_NAME
        config_path = out_dir / OUT_CONFIG_NAME
        bundle_path = out_dir / OUT_BUNDLE_NAME

        cm = metrics["confusion_matrix"]
        labels = metrics["labels"]
        test_counts = metrics["test_counts"]

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
            "training_augmentation": metrics.get("augmentation", {}),
            "cross_validation": {
                "enabled": bool(ENABLE_CROSS_VALIDATION),
                "strategy": str(CROSS_VALIDATION_STRATEGY),
                "groupkfold_n_splits": int(CV_GROUPKFOLD_N_SPLITS),
                "logo_max_groups": int(CV_LOGO_MAX_GROUPS),
                "cv_desc_used": metrics.get("cv_desc"),
            },

            "model": {
                "type": metrics.get("model"),
                "n_neighbors_requested": metrics.get("n_neighbors_requested"),
                "n_neighbors_used": metrics.get("n_neighbors_used"),
                "weights": metrics.get("weights_used"),
                "metric": metrics.get("metric_used"),
                "p": metrics.get("p_used"),
                "algorithm": metrics.get("algorithm_used"),
                "leaf_size": metrics.get("leaf_size_used"),
                "n_jobs": metrics.get("n_jobs"),
                "n_train_samples": metrics.get("n_train_samples"),
                "max_export_samples": metrics.get("max_export_samples", 0),
                "knn_search_debug": metrics.get("knn_search_debug", {}),
            },
            "dataset": {
                "data_root": str(data_root),
                "output_root": str(out_dir),
                "postures": list(selected_postures),
                "max_files_per_posture": max_files,
                "rows_per_posture": rows_per_posture,
                "rows_per_file": rows_per_file,
                "participant_split_numbers": {
                    "train": list(TRAIN_PARTICIPANT_NUMBERS),
                    "validation": list(VALIDATION_PARTICIPANT_NUMBERS),
                    "test": list(TEST_PARTICIPANT_NUMBERS),
                },
                "X_train_shape": list(X_train.shape),
                "X_val_shape": list(X_val.shape),
                "X_test_shape": list(X_test.shape),
                "seed": args.seed,
                "split_method": metrics["split_method"],
                "split_debug": metrics.get("split_debug", {}),
                "group_cnt_per_class_train": metrics.get("group_cnt_per_class_train", {}),
                "unique_groups_train": metrics.get("unique_groups_train", 0),
                "unique_groups_all": metrics.get("unique_groups_all", 0),
            },
            "outputs": {
                "header": OUT_HEADER_NAME,
                "config": OUT_CONFIG_NAME,
                "confusion_matrix": OUT_CM_NAME,
                "bundle": OUT_BUNDLE_NAME,
            },
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Config saved to: {config_path}")

        bundle: Dict[str, object] = {
            "model": metrics.get("model"),
            "classes": class_names,
            "feature_building": asdict(feat_cfg),
            "input_features": model_feature_names,
        }

        scaler: RobustScaler = pipeline.named_steps["scaler"]
        bundle["scaler_mean"] = scaler.center_.astype(np.float32)
        bundle["scaler_scale"] = scaler.scale_.astype(np.float32)

        clf: KNeighborsClassifier = pipeline.named_steps["clf"]

        # 记录超参(复现用)
        bundle["knn_params"] = {
            "n_neighbors": int(getattr(clf, "n_neighbors", 1)),
            "weights": str(getattr(clf, "weights", "uniform")),
            "metric": str(getattr(clf, "metric", "minkowski")),
            "p": int(getattr(clf, "p", 2) or 2),
            "algorithm": str(getattr(clf, "algorithm", "auto")),
            "leaf_size": int(getattr(clf, "leaf_size", KNN_LEAF_SIZE)),
            "n_jobs": int(getattr(clf, "n_jobs", KNN_N_JOBS) if getattr(clf, "n_jobs", None) is not None else KNN_N_JOBS),
            "max_export_samples": int(model_cfg.max_export_samples),
        }

        fit_X = np.asarray(getattr(clf, "_fit_X", np.zeros((0, len(model_feature_names)))), dtype=np.float32)
        fit_y_raw = np.asarray(getattr(clf, "_y", np.zeros((0,), dtype=np.int32))).reshape(-1)
        if np.issubdtype(fit_y_raw.dtype, np.integer):
            fit_y = fit_y_raw.astype(np.int32)
        else:
            class_to_idx = {c: i for i, c in enumerate(class_names)}
            fit_y = np.array([class_to_idx.get(v, 0) for v in fit_y_raw.tolist()], dtype=np.int32)

        bundle["knn_train_X"] = fit_X
        bundle["knn_train_y"] = fit_y
        bundle["knn_train_samples"] = int(fit_X.shape[0])

        joblib.dump(bundle, bundle_path)
        print(f"Model bundle saved to: {bundle_path}")

        print("\n================ 完成 ================")
        print("输出文件:")
        print(f"  {header_path}")
        print(f"  {config_path}")
        print(f"  {cm_path}")
        print(f"  {bundle_path}")


    else:
        print("\n================ 完成 ================")
        print("导出开关关闭: 已完成训练与评估(未写入任何输出文件).")
if __name__ == "__main__":
    main()
