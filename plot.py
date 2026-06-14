#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified plotting workflow.

This single script contains all plotting logic in one place.

Outputs:
1) svm_poly_accuracy_contour.jpg
2) svm_linear_rbf_accuracy.jpg
3) ann_accuracy_contour.jpg
4) tof_gated_weight_accuracy_comparison.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import AML_ANN as ann
import AML_DT as dt
import AML_KNN as knn
import AML_SVM as svm


SVM_POLY_DEGREE_GRID: List[int] = [2, 4, 6, 8, 10, 12, 14, 16]
SVM_C_GRID: List[float] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

ANN_NEURONS_GRID: List[int] = [16, 32, 64, 128, 256]
ANN_LAYERS_GRID: List[int] = [1, 2, 3, 4, 5]

TOF_WEIGHT_GRID: List[float] = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

CACHE_SCHEMA_VERSION = 1


DatasetSplit = Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    str,
    Dict[str, object],
]

AnnDatasetSplit = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all model plotting workflows in a single script."
    )
    parser.add_argument("--postures", type=str, default="", help="Comma-separated posture list.")
    parser.add_argument("--max_files_per_posture", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--rows_per_posture", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--rows_per_file", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--seed", type=int, default=int(svm.DEFAULT_SEED))
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument(
        "--svm_max_iter",
        type=int,
        default=10000,
        help="SVM max_iter for rbf/poly fits. <=0 keeps sklearn default.",
    )
    parser.add_argument(
        "--ann_device",
        type=str,
        default=str(ann.DEFAULT_ANN_DEVICE).strip().upper(),
        choices=["CPU", "GPU"],
        help="ANN training device.",
    )
    parser.add_argument(
        "--ann_verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="ANN training verbosity.",
    )
    parser.add_argument(
        "--knn_search_max_trials",
        type=int,
        default=60,
        help="KNN CV search trial cap for the single search round.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default="",
        help="Cache file path. Empty uses <output_dir>/plot_cache.pkl.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore cached values and recompute all sections.",
    )
    parser.add_argument(
        "--no_cache_use",
        action="store_true",
        help="Do not read cache.",
    )
    parser.add_argument(
        "--no_cache_write",
        action="store_true",
        help="Do not write cache.",
    )
    return parser.parse_args()


def _positive_or_none(v: int) -> int | None:
    return int(v) if int(v) > 0 else None


def _select_postures(detected_postures: Sequence[str], postures_arg: str) -> List[str]:
    if postures_arg.strip():
        selected = [p.strip() for p in postures_arg.split(",") if p.strip()]
    else:
        selected = list(detected_postures)
    missing = [p for p in selected if p not in set(detected_postures)]
    if missing:
        raise RuntimeError(f"Unknown postures: {missing}")
    return selected


def _normalize_for_signature(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_for_signature(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return repr(value)


def _make_signature(payload: Dict[str, Any]) -> str:
    normalized = _normalize_for_signature(payload)
    text = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _capture_module_constants(module: Any, *, names: Sequence[str], prefixes: Sequence[str]) -> Dict[str, Any]:
    module_vars = vars(module)
    result: Dict[str, Any] = {}
    for key, value in module_vars.items():
        if key.startswith("__"):
            continue
        if (key not in names) and (not any(key.startswith(p) for p in prefixes)):
            continue
        if callable(value):
            continue
        result[key] = _normalize_for_signature(value)
    return dict(sorted(result.items(), key=lambda kv: kv[0]))


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        with cache_path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print(f"[CACHE-WARN] failed to read cache {cache_path}: {exc}. Rebuilding cache.")
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}

    if not isinstance(data, dict):
        print(f"[CACHE-WARN] cache format invalid in {cache_path}. Rebuilding cache.")
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if int(data.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        print(
            "[CACHE-WARN] cache schema version mismatch "
            f"(found {data.get('schema_version')}, expected {CACHE_SCHEMA_VERSION}). Rebuilding cache."
        )
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    return data


def _save_cache(cache_path: Path, cache_data: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(cache_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)


def _get_or_compute(
    *,
    cache: Dict[str, Any],
    key: str,
    signature: str,
    compute_fn: Callable[[], Any],
    use_cache: bool,
    write_cache: bool,
    force_recompute: bool,
) -> Tuple[Any, bool]:
    entries = cache.setdefault("entries", {})
    entry = entries.get(key)

    if use_cache and (not force_recompute) and isinstance(entry, dict):
        if entry.get("signature") == signature:
            print(f"[CACHE-HIT] {key}")
            return entry.get("value"), False
        print(f"[CACHE-STALE] {key} signature changed, recomputing.")
    elif use_cache and (not force_recompute):
        print(f"[CACHE-MISS] {key}")
    elif force_recompute:
        print(f"[CACHE-BYPASS] {key} force recompute enabled.")
    else:
        print(f"[CACHE-BYPASS] {key} cache read disabled.")

    value = compute_fn()
    if write_cache:
        entries[key] = {"signature": signature, "value": value}
        print(f"[CACHE-SAVE] {key}")
        return value, True
    return value, False


def _prepare_dt_dataset(
    selected_postures: Sequence[str],
    max_files_per_posture: int | None,
    rows_per_posture: int | None,
    rows_per_file: int | None,
    seed: int,
) -> Tuple[DatasetSplit, List[str]]:
    data_root = Path(dt.DATA_ROOT_PATH).expanduser().resolve()
    feat_cfg = dt.FeatureBuildConfig(feature_keys=list(dt.SELECTED_FEATURE_KEYS))
    feature_names = dt.get_model_feature_names(feat_cfg.feature_keys)
    ds = dt.build_dataset_splits(
        data_root=data_root,
        postures=list(selected_postures),
        max_files_per_posture=max_files_per_posture,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=int(seed),
        feat_cfg=feat_cfg,
    )
    return ds, feature_names


def _prepare_svm_dataset(
    selected_postures: Sequence[str],
    max_files_per_posture: int | None,
    rows_per_posture: int | None,
    rows_per_file: int | None,
    seed: int,
) -> Tuple[DatasetSplit, List[str]]:
    data_root = Path(svm.DATA_ROOT_PATH).expanduser().resolve()
    feat_cfg = svm.FeatureBuildConfig(feature_keys=list(svm.SELECTED_FEATURE_KEYS))
    feature_names = svm.get_model_feature_names(feat_cfg.feature_keys)
    ds = svm.build_dataset_splits(
        data_root=data_root,
        postures=list(selected_postures),
        max_files_per_posture=max_files_per_posture,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=int(seed),
        feat_cfg=feat_cfg,
    )
    return ds, feature_names


def _prepare_knn_dataset(
    selected_postures: Sequence[str],
    max_files_per_posture: int | None,
    rows_per_posture: int | None,
    rows_per_file: int | None,
    seed: int,
) -> Tuple[DatasetSplit, List[str]]:
    data_root = Path(knn.DATA_ROOT_PATH).expanduser().resolve()
    feat_cfg = knn.FeatureBuildConfig(feature_keys=list(knn.SELECTED_FEATURE_KEYS))
    feature_names = knn.get_model_feature_names(feat_cfg.feature_keys)
    ds = knn.build_dataset_splits(
        data_root=data_root,
        postures=list(selected_postures),
        max_files_per_posture=max_files_per_posture,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=int(seed),
        feat_cfg=feat_cfg,
    )
    return ds, feature_names


def _prepare_ann_dataset(
    selected_postures: Sequence[str],
    max_files_per_posture: int | None,
    rows_per_posture: int | None,
    rows_per_file: int | None,
    seed: int,
) -> Tuple[AnnDatasetSplit, List[str]]:
    data_root = Path(ann.DATA_ROOT_PATH).expanduser().resolve()
    feat_cfg = ann.FeatureBuildConfig(feature_keys=list(ann.TRAIN_FEATURE_KEYS))
    feature_names = ann.get_model_feature_names(feat_cfg.feature_keys)

    (
        x_train,
        y_train,
        g_train,
        x_val,
        y_val,
        g_val,
        x_test,
        y_test,
        g_test,
        split_method,
        split_debug,
    ) = ann.build_dataset_splits(
        data_root=data_root,
        postures=list(selected_postures),
        max_files_per_posture=max_files_per_posture,
        rows_per_posture=rows_per_posture,
        rows_per_file=rows_per_file,
        seed=int(seed),
        feat_cfg=feat_cfg,
    )

    ds: AnnDatasetSplit = {
        "X_train": x_train,
        "y_train": y_train,
        "g_train": g_train,
        "X_val": x_val,
        "y_val": y_val,
        "g_val": g_val,
        "X_test": x_test,
        "y_test": y_test,
        "g_test": g_test,
        "split_method": split_method,
        "split_debug": split_debug,
    }
    return ds, feature_names


def _grid_levels(z: np.ndarray, n_levels: int = 12) -> np.ndarray:
    z_valid = z[np.isfinite(z)]
    if z_valid.size == 0:
        raise RuntimeError("No finite accuracy values available for plotting.")
    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))
    if math.isclose(z_min, z_max, rel_tol=1e-12, abs_tol=1e-12):
        eps = max(1e-6, abs(z_min) * 1e-3)
        return np.linspace(z_min - eps, z_max + eps, 3)
    return np.linspace(z_min, z_max, n_levels)


def save_contour_plot(
    x_values: Sequence[float],
    y_values: Sequence[float],
    z_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    y_log_scale: bool = False,
) -> None:
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    z = np.array(z_values, dtype=float)
    xx, yy = np.meshgrid(x, y)

    levels = _grid_levels(z)
    z_masked = np.ma.masked_invalid(z)

    fig, ax = plt.subplots(figsize=(10, 7))
    cf = ax.contourf(xx, yy, z_masked, levels=levels, cmap="viridis")
    c = ax.contour(xx, yy, z_masked, levels=levels, colors="white", linewidths=0.8, alpha=0.85)
    if len(c.levels) > 1:
        ax.clabel(c, inline=True, fontsize=8, fmt="%.3f")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Accuracy")
    if y_log_scale:
        ax.set_yscale("log")
    ax.set_xticks(list(x_values))
    ax.set_yticks(list(y_values))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, format="jpeg", dpi=300)
    plt.close(fig)


def save_linear_rbf_plot(
    c_values: Sequence[float],
    linear_acc: Sequence[float],
    rbf_acc: Sequence[float],
    out_path: Path,
) -> None:
    x = np.array(c_values, dtype=float)
    y_linear = np.array(linear_acc, dtype=float)
    y_rbf = np.array(rbf_acc, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_linear, marker="o", linewidth=2.0, label="SVM Linear")
    ax.plot(x, y_rbf, marker="s", linewidth=2.0, label="SVM RBF")
    ax.set_xscale("log")
    ax.set_xticks(list(c_values))
    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.set_title("SVM Linear vs RBF Accuracy Across C")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="jpeg", dpi=300)
    plt.close(fig)


def save_tof_line_plot(
    x_values: Sequence[float],
    curves: Dict[str, Sequence[float]],
    legend_labels: Dict[str, str],
    out_path: Path,
) -> None:
    x = np.array(x_values, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))
    order = ["svm_poly", "svm_rbf", "svm_linear", "knn", "dt", "ann"]
    styles = {
        "svm_poly": dict(marker="o", linewidth=2.0),
        "svm_rbf": dict(marker="s", linewidth=2.0),
        "svm_linear": dict(marker="^", linewidth=2.0),
        "knn": dict(marker="D", linewidth=2.0),
        "dt": dict(marker="P", linewidth=2.0),
        "ann": dict(marker="X", linewidth=2.0),
    }

    for k in order:
        y = np.array(curves[k], dtype=float)
        ax.plot(x, y, label=legend_labels[k], **styles[k])

    ax.set_xlabel("TOF_GATED_WEIGHT")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs TOF_GATED_WEIGHT (Best Other Hyper-Parameters Fixed)")
    ax.set_xticks(list(x_values))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="jpeg", dpi=300)
    plt.close(fig)


def evaluate_dt_best_by_search(
    dataset: DatasetSplit,
    feature_names: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    (x_train, y_train, g_train), (x_val, y_val, g_val), (x_test, y_test, g_test), split_method, split_debug = dataset

    original_cv_enabled = bool(dt.ENABLE_CROSS_VALIDATION)
    dt.ENABLE_CROSS_VALIDATION = True
    try:
        model_cfg = dt.ModelConfig(
            criterion=dt.DT_CRITERION,
            splitter=dt.DT_SPLITTER,
            max_depth=dt.DT_MAX_DEPTH,
            min_samples_split=int(dt.DT_MIN_SAMPLES_SPLIT),
            min_samples_leaf=int(dt.DT_MIN_SAMPLES_LEAF),
            max_features=dt.DT_MAX_FEATURES,
            max_leaf_nodes=dt.DT_MAX_LEAF_NODES,
            min_impurity_decrease=dt.DT_MIN_IMPURITY_DECREASE,
            ccp_alpha=dt.DT_CCP_ALPHA,
            class_weight=dt.DEFAULT_CLASS_WEIGHT,
            max_nodes=dt.DT_MAX_NODES,
        )
        _, metrics, _ = dt.train_and_evaluate(
            X_train=x_train,
            y_train=y_train,
            groups_train=g_train,
            X_val=x_val,
            y_val=y_val,
            groups_val=g_val,
            X_test=x_test,
            y_test=y_test,
            groups_test=g_test,
            feature_names=list(feature_names),
            model_cfg=model_cfg,
            seed=int(seed),
            split_method=split_method,
            split_debug=split_debug,
        )
    finally:
        dt.ENABLE_CROSS_VALIDATION = original_cv_enabled

    best = {
        "criterion": str(metrics.get("criterion_used")),
        "splitter": str(metrics.get("splitter_used")),
        "max_depth": metrics.get("max_depth_used"),
        "min_samples_split": int(metrics.get("min_samples_split_used")),
        "min_samples_leaf": int(metrics.get("min_samples_leaf_used")),
        "max_features": metrics.get("max_features_used"),
        "max_leaf_nodes": metrics.get("max_leaf_nodes_used"),
        "min_impurity_decrease": float(metrics.get("min_impurity_decrease_used")),
        "ccp_alpha": float(metrics.get("ccp_alpha_used")),
        "accuracy": float(metrics.get("accuracy")),
    }
    print(
        "[DT-SEARCH-BEST] "
        f"criterion={best['criterion']}, splitter={best['splitter']}, "
        f"min_samples_split={best['min_samples_split']}, min_samples_leaf={best['min_samples_leaf']}, "
        f"max_depth={best['max_depth']}, accuracy={best['accuracy']:.6f}"
    )
    return best


def evaluate_svm_poly_grid(
    dataset: DatasetSplit,
    feature_names: Sequence[str],
    seed: int,
) -> np.ndarray:
    (x_train, y_train, g_train), (x_val, y_val, g_val), (x_test, y_test, g_test), split_method, split_debug = dataset

    z = np.full((len(SVM_C_GRID), len(SVM_POLY_DEGREE_GRID)), np.nan, dtype=float)
    for yi, c_val in enumerate(SVM_C_GRID):
        for xi, degree in enumerate(SVM_POLY_DEGREE_GRID):
            cfg = svm.ModelConfig(
                kernel="poly",
                C=float(c_val),
                gamma=svm.DEFAULT_GAMMA,
                degree=int(degree),
                coef0=float(svm.DEFAULT_POLY_COEF0),
                class_weight=svm.DEFAULT_CLASS_WEIGHT,
                sv_max=int(svm.SV_MAX_COUNT),
            )
            _, metrics, _ = svm.train_and_evaluate(
                X_train=x_train,
                y_train=y_train,
                groups_train=g_train,
                X_val=x_val,
                y_val=y_val,
                groups_val=g_val,
                X_test=x_test,
                y_test=y_test,
                groups_test=g_test,
                feature_names=list(feature_names),
                model_cfg=cfg,
                seed=int(seed),
                split_method=split_method,
                split_debug=split_debug,
            )
            z[yi, xi] = float(metrics["accuracy"])
            print(f"[SVM-POLY] degree={degree:>2}, C={c_val:<8g}, accuracy={z[yi, xi]:.6f}")
    return z


def evaluate_svm_linear_rbf(
    dataset: DatasetSplit,
    feature_names: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    (x_train, y_train, g_train), (x_val, y_val, g_val), (x_test, y_test, g_test), split_method, split_debug = dataset

    linear_acc = np.full((len(SVM_C_GRID),), np.nan, dtype=float)
    rbf_acc = np.full((len(SVM_C_GRID),), np.nan, dtype=float)

    for i, c_val in enumerate(SVM_C_GRID):
        linear_cfg = svm.ModelConfig(
            kernel="linear",
            C=float(c_val),
            gamma=svm.DEFAULT_GAMMA,
            degree=int(svm.DEFAULT_POLY_DEGREE),
            coef0=float(svm.DEFAULT_POLY_COEF0),
            class_weight=svm.DEFAULT_CLASS_WEIGHT,
            sv_max=int(svm.SV_MAX_COUNT),
        )
        _, linear_metrics, _ = svm.train_and_evaluate(
            X_train=x_train,
            y_train=y_train,
            groups_train=g_train,
            X_val=x_val,
            y_val=y_val,
            groups_val=g_val,
            X_test=x_test,
            y_test=y_test,
            groups_test=g_test,
            feature_names=list(feature_names),
            model_cfg=linear_cfg,
            seed=int(seed),
            split_method=split_method,
            split_debug=split_debug,
        )
        linear_acc[i] = float(linear_metrics["accuracy"])

        rbf_cfg = svm.ModelConfig(
            kernel="rbf",
            C=float(c_val),
            gamma=svm.DEFAULT_GAMMA,
            degree=int(svm.DEFAULT_POLY_DEGREE),
            coef0=float(svm.DEFAULT_POLY_COEF0),
            class_weight=svm.DEFAULT_CLASS_WEIGHT,
            sv_max=int(svm.SV_MAX_COUNT),
        )
        _, rbf_metrics, _ = svm.train_and_evaluate(
            X_train=x_train,
            y_train=y_train,
            groups_train=g_train,
            X_val=x_val,
            y_val=y_val,
            groups_val=g_val,
            X_test=x_test,
            y_test=y_test,
            groups_test=g_test,
            feature_names=list(feature_names),
            model_cfg=rbf_cfg,
            seed=int(seed),
            split_method=split_method,
            split_debug=split_debug,
        )
        rbf_acc[i] = float(rbf_metrics["accuracy"])

        print(
            f"[SVM-LINEAR/RBF] C={c_val:<8g}, linear_acc={linear_acc[i]:.6f}, rbf_acc={rbf_acc[i]:.6f}"
        )

    return linear_acc, rbf_acc


def best_svm_from_results(
    poly_z: np.ndarray,
    linear_acc: np.ndarray,
    rbf_acc: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    poly_idx = int(np.nanargmax(poly_z))
    poly_yi, poly_xi = np.unravel_index(poly_idx, poly_z.shape)

    linear_i = int(np.nanargmax(linear_acc))
    rbf_i = int(np.nanargmax(rbf_acc))

    return {
        "svm_poly": {
            "degree": int(SVM_POLY_DEGREE_GRID[poly_xi]),
            "C": float(SVM_C_GRID[poly_yi]),
            "accuracy": float(poly_z[poly_yi, poly_xi]),
        },
        "svm_linear": {
            "C": float(SVM_C_GRID[linear_i]),
            "accuracy": float(linear_acc[linear_i]),
        },
        "svm_rbf": {
            "C": float(SVM_C_GRID[rbf_i]),
            "accuracy": float(rbf_acc[rbf_i]),
        },
    }


def evaluate_ann_grid(
    ann_ds: AnnDatasetSplit,
    feature_names: Sequence[str],
    ann_device: str,
    ann_verbose: int,
    seed: int,
) -> np.ndarray:
    z = np.full((len(ANN_LAYERS_GRID), len(ANN_NEURONS_GRID)), np.nan, dtype=float)

    for yi, layer_cnt in enumerate(ANN_LAYERS_GRID):
        for xi, neurons in enumerate(ANN_NEURONS_GRID):
            hidden_layers = [int(neurons)] * int(layer_cnt)
            model_cfg = ann.ModelConfig(
                hidden_layers=hidden_layers,
                activation=ann.DEFAULT_ANN_ACTIVATION,
                dropout=float(ann.DEFAULT_ANN_DROPOUT),
                tof_gated_weight=float(ann.TOF_GATED_WEIGHT),
                batch_size=int(ann.DEFAULT_ANN_BATCH_SIZE),
                epochs=int(ann.DEFAULT_ANN_EPOCHS),
                lr=float(ann.DEFAULT_ANN_LR),
                weight_decay=float(ann.DEFAULT_ANN_WEIGHT_DECAY),
                lr_scheduler=str(ann.DEFAULT_ANN_LR_SCHEDULER),
                lr_min=float(ann.DEFAULT_ANN_LR_MIN),
                lr_plateau_factor=float(ann.DEFAULT_ANN_LR_PLATEAU_FACTOR),
                lr_plateau_patience=int(ann.DEFAULT_ANN_LR_PLATEAU_PATIENCE),
                lr_plateau_threshold=float(ann.DEFAULT_ANN_LR_PLATEAU_THRESHOLD),
                lr_plateau_cooldown=int(ann.DEFAULT_ANN_LR_PLATEAU_COOLDOWN),
                wd_scheduler=str(ann.DEFAULT_ANN_WD_SCHEDULER),
                weight_decay_end=float(ann.DEFAULT_ANN_WEIGHT_DECAY_END),
                early_stop_patience=int(ann.DEFAULT_ANN_EARLY_STOP_PATIENCE),
                min_delta=float(ann.DEFAULT_ANN_MIN_DELTA),
                use_class_weights=bool(ann.DEFAULT_ANN_USE_CLASS_WEIGHTS),
                device=str(ann_device).upper(),
                verbose=int(ann_verbose),
                random_state=int(seed),
            )
            _, metrics, _ = ann.train_and_evaluate(
                feature_names=list(feature_names),
                X_train=ann_ds["X_train"],
                y_train=ann_ds["y_train"],
                g_train=ann_ds["g_train"],
                X_val=ann_ds["X_val"],
                y_val=ann_ds["y_val"],
                g_val=ann_ds["g_val"],
                X_test=ann_ds["X_test"],
                y_test=ann_ds["y_test"],
                g_test=ann_ds["g_test"],
                model_cfg=model_cfg,
                seed=int(seed),
                split_method=str(ann_ds["split_method"]),
                split_debug=dict(ann_ds["split_debug"]),
            )
            z[yi, xi] = float(metrics["accuracy"])
            print(f"[ANN] layers={layer_cnt}, neurons_per_layer={neurons}, accuracy={z[yi, xi]:.6f}")
    return z


def best_ann_from_grid(ann_z: np.ndarray) -> Dict[str, Any]:
    idx = int(np.nanargmax(ann_z))
    yi, xi = np.unravel_index(idx, ann_z.shape)
    layers = int(ANN_LAYERS_GRID[yi])
    neurons = int(ANN_NEURONS_GRID[xi])
    return {
        "layers": layers,
        "neurons_per_layer": neurons,
        "hidden_layers": [neurons] * layers,
        "accuracy": float(ann_z[yi, xi]),
    }


def run_knn_search_once(
    knn_ds: DatasetSplit,
    feature_names: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    (x_train, y_train, g_train), (x_val, y_val, g_val), (x_test, y_test, g_test), split_method, split_debug = knn_ds

    model_cfg = knn.ModelConfig(
        n_neighbors=int(knn.KNN_N_NEIGHBORS),
        weights=str(knn.KNN_WEIGHTS),
        metric=str(knn.KNN_METRIC),
        p=int(knn.KNN_P),
        algorithm=str(knn.KNN_ALGORITHM),
        leaf_size=int(knn.KNN_LEAF_SIZE),
        n_jobs=int(knn.KNN_N_JOBS),
        max_export_samples=int(knn.KNN_MAX_EXPORT_SAMPLES),
    )
    _, metrics, _ = knn.train_and_evaluate(
        X_train=x_train,
        y_train=y_train,
        groups_train=g_train,
        X_val=x_val,
        y_val=y_val,
        groups_val=g_val,
        X_test=x_test,
        y_test=y_test,
        groups_test=g_test,
        feature_names=list(feature_names),
        model_cfg=model_cfg,
        seed=int(seed),
        split_method=split_method,
        split_debug=split_debug,
    )

    best = {
        "n_neighbors": int(metrics["n_neighbors_requested"]),
        "weights": str(metrics["weights_used"]),
        "metric": str(metrics["metric_used"]),
        "p": int(metrics["p_used"]),
        "algorithm": str(metrics["algorithm_used"]),
        "leaf_size": int(metrics["leaf_size_used"]),
        "accuracy": float(metrics["accuracy"]),
    }
    print(
        "[KNN-SEARCH-BEST] "
        f"k={best['n_neighbors']}, weights={best['weights']}, metric={best['metric']}, "
        f"p={best['p']}, algorithm={best['algorithm']}, leaf_size={best['leaf_size']}, "
        f"accuracy={best['accuracy']:.6f}"
    )
    return best


def evaluate_tof_weight_curves(
    dt_best: Dict[str, Any],
    svm_best: Dict[str, Dict[str, Any]],
    knn_best: Dict[str, Any],
    ann_best: Dict[str, Any],
    dt_ds: DatasetSplit,
    dt_feature_names: Sequence[str],
    svm_ds: DatasetSplit,
    svm_feature_names: Sequence[str],
    knn_ds: DatasetSplit,
    knn_feature_names: Sequence[str],
    ann_ds: AnnDatasetSplit,
    ann_feature_names: Sequence[str],
    ann_device: str,
    ann_verbose: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    curves = {
        "tof_dt": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
        "tof_svm_poly": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
        "tof_svm_rbf": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
        "tof_svm_linear": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
        "tof_knn": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
        "tof_ann": np.full((len(TOF_WEIGHT_GRID),), np.nan, dtype=float),
    }

    (dt_x_train, dt_y_train, dt_g_train), (dt_x_val, dt_y_val, dt_g_val), (dt_x_test, dt_y_test, dt_g_test), dt_split_method, dt_split_debug = dt_ds
    (svm_x_train, svm_y_train, svm_g_train), (svm_x_val, svm_y_val, svm_g_val), (svm_x_test, svm_y_test, svm_g_test), svm_split_method, svm_split_debug = svm_ds
    (knn_x_train, knn_y_train, knn_g_train), (knn_x_val, knn_y_val, knn_g_val), (knn_x_test, knn_y_test, knn_g_test), knn_split_method, knn_split_debug = knn_ds

    for i, tof_weight in enumerate(TOF_WEIGHT_GRID):
        dt.TOF_GATED_WEIGHT = float(tof_weight)
        svm.TOF_GATED_WEIGHT = float(tof_weight)
        knn.TOF_GATED_WEIGHT = float(tof_weight)
        ann.TOF_GATED_WEIGHT = float(tof_weight)

        dt_cfg = dt.ModelConfig(
            criterion=str(dt_best.get("criterion", dt.DT_CRITERION)),
            splitter=str(dt_best.get("splitter", dt.DT_SPLITTER)),
            max_depth=dt_best.get("max_depth", dt.DT_MAX_DEPTH),
            min_samples_split=int(dt_best["min_samples_split"]),
            min_samples_leaf=int(dt_best["min_samples_leaf"]),
            max_features=dt_best.get("max_features", dt.DT_MAX_FEATURES),
            max_leaf_nodes=dt_best.get("max_leaf_nodes", dt.DT_MAX_LEAF_NODES),
            min_impurity_decrease=float(dt_best.get("min_impurity_decrease", dt.DT_MIN_IMPURITY_DECREASE)),
            ccp_alpha=float(dt_best.get("ccp_alpha", dt.DT_CCP_ALPHA)),
            class_weight=dt.DEFAULT_CLASS_WEIGHT,
            max_nodes=dt.DT_MAX_NODES,
        )
        _, dt_metrics, _ = dt.train_and_evaluate(
            X_train=dt_x_train,
            y_train=dt_y_train,
            groups_train=dt_g_train,
            X_val=dt_x_val,
            y_val=dt_y_val,
            groups_val=dt_g_val,
            X_test=dt_x_test,
            y_test=dt_y_test,
            groups_test=dt_g_test,
            feature_names=list(dt_feature_names),
            model_cfg=dt_cfg,
            seed=int(seed),
            split_method=dt_split_method,
            split_debug=dt_split_debug,
        )
        curves["tof_dt"][i] = float(dt_metrics["accuracy"])

        svm_poly_cfg = svm.ModelConfig(
            kernel="poly",
            C=float(svm_best["svm_poly"]["C"]),
            gamma=svm.DEFAULT_GAMMA,
            degree=int(svm_best["svm_poly"]["degree"]),
            coef0=float(svm.DEFAULT_POLY_COEF0),
            class_weight=svm.DEFAULT_CLASS_WEIGHT,
            sv_max=int(svm.SV_MAX_COUNT),
        )
        _, svm_poly_metrics, _ = svm.train_and_evaluate(
            X_train=svm_x_train,
            y_train=svm_y_train,
            groups_train=svm_g_train,
            X_val=svm_x_val,
            y_val=svm_y_val,
            groups_val=svm_g_val,
            X_test=svm_x_test,
            y_test=svm_y_test,
            groups_test=svm_g_test,
            feature_names=list(svm_feature_names),
            model_cfg=svm_poly_cfg,
            seed=int(seed),
            split_method=svm_split_method,
            split_debug=svm_split_debug,
        )
        curves["tof_svm_poly"][i] = float(svm_poly_metrics["accuracy"])

        svm_rbf_cfg = svm.ModelConfig(
            kernel="rbf",
            C=float(svm_best["svm_rbf"]["C"]),
            gamma=svm.DEFAULT_GAMMA,
            degree=int(svm.DEFAULT_POLY_DEGREE),
            coef0=float(svm.DEFAULT_POLY_COEF0),
            class_weight=svm.DEFAULT_CLASS_WEIGHT,
            sv_max=int(svm.SV_MAX_COUNT),
        )
        _, svm_rbf_metrics, _ = svm.train_and_evaluate(
            X_train=svm_x_train,
            y_train=svm_y_train,
            groups_train=svm_g_train,
            X_val=svm_x_val,
            y_val=svm_y_val,
            groups_val=svm_g_val,
            X_test=svm_x_test,
            y_test=svm_y_test,
            groups_test=svm_g_test,
            feature_names=list(svm_feature_names),
            model_cfg=svm_rbf_cfg,
            seed=int(seed),
            split_method=svm_split_method,
            split_debug=svm_split_debug,
        )
        curves["tof_svm_rbf"][i] = float(svm_rbf_metrics["accuracy"])

        svm_linear_cfg = svm.ModelConfig(
            kernel="linear",
            C=float(svm_best["svm_linear"]["C"]),
            gamma=svm.DEFAULT_GAMMA,
            degree=int(svm.DEFAULT_POLY_DEGREE),
            coef0=float(svm.DEFAULT_POLY_COEF0),
            class_weight=svm.DEFAULT_CLASS_WEIGHT,
            sv_max=int(svm.SV_MAX_COUNT),
        )
        _, svm_linear_metrics, _ = svm.train_and_evaluate(
            X_train=svm_x_train,
            y_train=svm_y_train,
            groups_train=svm_g_train,
            X_val=svm_x_val,
            y_val=svm_y_val,
            groups_val=svm_g_val,
            X_test=svm_x_test,
            y_test=svm_y_test,
            groups_test=svm_g_test,
            feature_names=list(svm_feature_names),
            model_cfg=svm_linear_cfg,
            seed=int(seed),
            split_method=svm_split_method,
            split_debug=svm_split_debug,
        )
        curves["tof_svm_linear"][i] = float(svm_linear_metrics["accuracy"])

        knn_cfg = knn.ModelConfig(
            n_neighbors=int(knn_best["n_neighbors"]),
            weights=str(knn_best["weights"]),
            metric=str(knn_best["metric"]),
            p=int(knn_best["p"]),
            algorithm=str(knn_best["algorithm"]),
            leaf_size=int(knn_best["leaf_size"]),
            n_jobs=int(knn.KNN_N_JOBS),
            max_export_samples=int(knn.KNN_MAX_EXPORT_SAMPLES),
        )
        _, knn_metrics, _ = knn.train_and_evaluate(
            X_train=knn_x_train,
            y_train=knn_y_train,
            groups_train=knn_g_train,
            X_val=knn_x_val,
            y_val=knn_y_val,
            groups_val=knn_g_val,
            X_test=knn_x_test,
            y_test=knn_y_test,
            groups_test=knn_g_test,
            feature_names=list(knn_feature_names),
            model_cfg=knn_cfg,
            seed=int(seed),
            split_method=knn_split_method,
            split_debug=knn_split_debug,
        )
        curves["tof_knn"][i] = float(knn_metrics["accuracy"])

        ann_cfg = ann.ModelConfig(
            hidden_layers=list(ann_best["hidden_layers"]),
            activation=ann.DEFAULT_ANN_ACTIVATION,
            dropout=float(ann.DEFAULT_ANN_DROPOUT),
            tof_gated_weight=float(tof_weight),
            batch_size=int(ann.DEFAULT_ANN_BATCH_SIZE),
            epochs=int(ann.DEFAULT_ANN_EPOCHS),
            lr=float(ann.DEFAULT_ANN_LR),
            weight_decay=float(ann.DEFAULT_ANN_WEIGHT_DECAY),
            lr_scheduler=str(ann.DEFAULT_ANN_LR_SCHEDULER),
            lr_min=float(ann.DEFAULT_ANN_LR_MIN),
            lr_plateau_factor=float(ann.DEFAULT_ANN_LR_PLATEAU_FACTOR),
            lr_plateau_patience=int(ann.DEFAULT_ANN_LR_PLATEAU_PATIENCE),
            lr_plateau_threshold=float(ann.DEFAULT_ANN_LR_PLATEAU_THRESHOLD),
            lr_plateau_cooldown=int(ann.DEFAULT_ANN_LR_PLATEAU_COOLDOWN),
            wd_scheduler=str(ann.DEFAULT_ANN_WD_SCHEDULER),
            weight_decay_end=float(ann.DEFAULT_ANN_WEIGHT_DECAY_END),
            early_stop_patience=int(ann.DEFAULT_ANN_EARLY_STOP_PATIENCE),
            min_delta=float(ann.DEFAULT_ANN_MIN_DELTA),
            use_class_weights=bool(ann.DEFAULT_ANN_USE_CLASS_WEIGHTS),
            device=str(ann_device).upper(),
            verbose=int(ann_verbose),
            random_state=int(seed),
        )
        _, ann_metrics, _ = ann.train_and_evaluate(
            feature_names=list(ann_feature_names),
            X_train=ann_ds["X_train"],
            y_train=ann_ds["y_train"],
            g_train=ann_ds["g_train"],
            X_val=ann_ds["X_val"],
            y_val=ann_ds["y_val"],
            g_val=ann_ds["g_val"],
            X_test=ann_ds["X_test"],
            y_test=ann_ds["y_test"],
            g_test=ann_ds["g_test"],
            model_cfg=ann_cfg,
            seed=int(seed),
            split_method=str(ann_ds["split_method"]),
            split_debug=dict(ann_ds["split_debug"]),
        )
        curves["tof_ann"][i] = float(ann_metrics["accuracy"])

        print(
            f"[TOF] weight={tof_weight:<4g} "
            f"svm_poly={curves['tof_svm_poly'][i]:.6f}, svm_rbf={curves['tof_svm_rbf'][i]:.6f}, "
            f"svm_linear={curves['tof_svm_linear'][i]:.6f}, knn={curves['tof_knn'][i]:.6f}, "
            f"dt={curves['tof_dt'][i]:.6f}, ann={curves['tof_ann'][i]:.6f}"
        )

    return curves


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_files_per_posture = _positive_or_none(args.max_files_per_posture)
    rows_per_posture = _positive_or_none(args.rows_per_posture)
    rows_per_file = _positive_or_none(args.rows_per_file)
    seed = int(args.seed)
    ann_device = str(args.ann_device).upper()
    ann_verbose = int(args.ann_verbose)
    use_cache = not bool(args.no_cache_use)
    write_cache = not bool(args.no_cache_write)
    force_recompute = bool(args.force_recompute)
    cache_path = (
        Path(args.cache_file).expanduser().resolve()
        if str(args.cache_file).strip()
        else (output_dir / "plot_cache.pkl")
    )

    dt_root = Path(dt.DATA_ROOT_PATH).expanduser().resolve()
    detected_postures = dt.auto_detect_postures_folder_only(dt_root)
    if not detected_postures:
        raise RuntimeError(f"No posture folders found under {dt_root}")
    selected_postures = _select_postures(detected_postures, args.postures)

    # Force exact evaluation for final curves.
    dt.ENABLE_CROSS_VALIDATION = False
    svm.ENABLE_CROSS_VALIDATION = False
    knn.ENABLE_CROSS_VALIDATION = False
    if int(args.knn_search_max_trials) > 0:
        knn.KNN_SEARCH_MAX_TRIALS = int(args.knn_search_max_trials)

    if int(args.svm_max_iter) > 0:
        _svc_cls = svm.SVC

        def _svc_with_iter_cap(*svc_args, **svc_kwargs):
            svc_kwargs.setdefault("max_iter", int(args.svm_max_iter))
            return _svc_cls(*svc_args, **svc_kwargs)

        svm.SVC = _svc_with_iter_cap

    cache = _load_cache(cache_path) if use_cache else {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    cache_changed = False

    print(f"Selected postures: {selected_postures}")
    print(f"Output directory: {output_dir}")
    print(f"SVM max_iter cap: {int(args.svm_max_iter)}")
    print(f"KNN search max trials: {int(knn.KNN_SEARCH_MAX_TRIALS)}")
    print(
        f"Cache settings: path={cache_path}, read={'on' if use_cache else 'off'}, "
        f"write={'on' if write_cache else 'off'}, force_recompute={force_recompute}"
    )

    datasets: Dict[str, Any] = {}

    def get_dt_dataset() -> Tuple[DatasetSplit, List[str]]:
        if "dt" not in datasets:
            print("[DATASET] building DT dataset...")
            datasets["dt"] = _prepare_dt_dataset(
                selected_postures=selected_postures,
                max_files_per_posture=max_files_per_posture,
                rows_per_posture=rows_per_posture,
                rows_per_file=rows_per_file,
                seed=seed,
            )
        return datasets["dt"]

    def get_svm_dataset() -> Tuple[DatasetSplit, List[str]]:
        if "svm" not in datasets:
            print("[DATASET] building SVM dataset...")
            datasets["svm"] = _prepare_svm_dataset(
                selected_postures=selected_postures,
                max_files_per_posture=max_files_per_posture,
                rows_per_posture=rows_per_posture,
                rows_per_file=rows_per_file,
                seed=seed,
            )
        return datasets["svm"]

    def get_knn_dataset() -> Tuple[DatasetSplit, List[str]]:
        if "knn" not in datasets:
            print("[DATASET] building KNN dataset...")
            datasets["knn"] = _prepare_knn_dataset(
                selected_postures=selected_postures,
                max_files_per_posture=max_files_per_posture,
                rows_per_posture=rows_per_posture,
                rows_per_file=rows_per_file,
                seed=seed,
            )
        return datasets["knn"]

    def get_ann_dataset() -> Tuple[AnnDatasetSplit, List[str]]:
        if "ann" not in datasets:
            print("[DATASET] building ANN dataset...")
            datasets["ann"] = _prepare_ann_dataset(
                selected_postures=selected_postures,
                max_files_per_posture=max_files_per_posture,
                rows_per_posture=rows_per_posture,
                rows_per_file=rows_per_file,
                seed=seed,
            )
        return datasets["ann"]

    run_context = {
        "seed": seed,
        "postures": list(selected_postures),
        "max_files_per_posture": max_files_per_posture,
        "rows_per_posture": rows_per_posture,
        "rows_per_file": rows_per_file,
        "svm_max_iter": int(args.svm_max_iter),
        "knn_search_max_trials": int(knn.KNN_SEARCH_MAX_TRIALS),
        "ann_device": ann_device,
        "ann_verbose": ann_verbose,
    }
    dt_consts = _capture_module_constants(
        dt,
        names=("SELECTED_FEATURE_KEYS",),
        prefixes=("DT_", "TOF_", "DEFAULT_", "DATA_ROOT_PATH"),
    )
    svm_consts = _capture_module_constants(
        svm,
        names=("SELECTED_FEATURE_KEYS",),
        prefixes=("DEFAULT_", "SVM_", "SV_", "TOF_", "DATA_ROOT_PATH"),
    )
    knn_consts = _capture_module_constants(
        knn,
        names=("SELECTED_FEATURE_KEYS",),
        prefixes=("KNN_", "TOF_", "DEFAULT_", "DATA_ROOT_PATH"),
    )
    ann_consts = _capture_module_constants(
        ann,
        names=("TRAIN_FEATURE_KEYS",),
        prefixes=("DEFAULT_ANN_", "TOF_", "DATA_ROOT_PATH"),
    )

    dt_best_sig = _make_signature(
        {
            "section": "dt_best",
            "context": run_context,
            "constants": dt_consts,
        }
    )
    dt_best, changed = _get_or_compute(
        cache=cache,
        key="dt_best",
        signature=dt_best_sig,
        compute_fn=lambda: evaluate_dt_best_by_search(*get_dt_dataset(), seed=seed),
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    cache_changed = cache_changed or changed

    svm_poly_sig = _make_signature(
        {
            "section": "svm_poly_grid",
            "context": run_context,
            "constants": svm_consts,
            "svm_poly_degree_grid": SVM_POLY_DEGREE_GRID,
            "svm_c_grid": SVM_C_GRID,
        }
    )
    poly_z, changed = _get_or_compute(
        cache=cache,
        key="svm_poly_grid",
        signature=svm_poly_sig,
        compute_fn=lambda: evaluate_svm_poly_grid(*get_svm_dataset(), seed=seed),
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    poly_z = np.array(poly_z, dtype=float)
    cache_changed = cache_changed or changed

    svm_linear_rbf_sig = _make_signature(
        {
            "section": "svm_linear_rbf",
            "context": run_context,
            "constants": svm_consts,
            "svm_c_grid": SVM_C_GRID,
        }
    )

    def _compute_svm_linear_rbf_bundle() -> Dict[str, np.ndarray]:
        linear_vals, rbf_vals = evaluate_svm_linear_rbf(*get_svm_dataset(), seed=seed)
        return {"linear_acc": linear_vals, "rbf_acc": rbf_vals}

    linear_rbf_bundle, changed = _get_or_compute(
        cache=cache,
        key="svm_linear_rbf",
        signature=svm_linear_rbf_sig,
        compute_fn=_compute_svm_linear_rbf_bundle,
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    linear_acc = np.array(linear_rbf_bundle["linear_acc"], dtype=float)
    rbf_acc = np.array(linear_rbf_bundle["rbf_acc"], dtype=float)
    cache_changed = cache_changed or changed

    svm_best = best_svm_from_results(poly_z, linear_acc, rbf_acc)
    print(
        "[SVM-BEST] "
        f"poly(deg={svm_best['svm_poly']['degree']}, C={svm_best['svm_poly']['C']}, acc={svm_best['svm_poly']['accuracy']:.6f}), "
        f"rbf(C={svm_best['svm_rbf']['C']}, acc={svm_best['svm_rbf']['accuracy']:.6f}), "
        f"linear(C={svm_best['svm_linear']['C']}, acc={svm_best['svm_linear']['accuracy']:.6f})"
    )

    ann_grid_sig = _make_signature(
        {
            "section": "ann_grid",
            "context": run_context,
            "constants": ann_consts,
            "ann_neurons_grid": ANN_NEURONS_GRID,
            "ann_layers_grid": ANN_LAYERS_GRID,
        }
    )
    ann_z, changed = _get_or_compute(
        cache=cache,
        key="ann_grid",
        signature=ann_grid_sig,
        compute_fn=lambda: evaluate_ann_grid(
            ann_ds=get_ann_dataset()[0],
            feature_names=get_ann_dataset()[1],
            ann_device=ann_device,
            ann_verbose=ann_verbose,
            seed=seed,
        ),
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    ann_z = np.array(ann_z, dtype=float)
    cache_changed = cache_changed or changed

    ann_best = best_ann_from_grid(ann_z)
    print(
        "[ANN-BEST] "
        f"layers={ann_best['layers']}, neurons_per_layer={ann_best['neurons_per_layer']}, "
        f"accuracy={ann_best['accuracy']:.6f}"
    )

    def _compute_knn_best() -> Dict[str, Any]:
        print("[KNN-SEARCH] running one round of KNN hyper-parameter search...")
        original_knn_cv = bool(knn.ENABLE_CROSS_VALIDATION)
        knn.ENABLE_CROSS_VALIDATION = True
        try:
            return run_knn_search_once(
                knn_ds=get_knn_dataset()[0],
                feature_names=get_knn_dataset()[1],
                seed=seed,
            )
        finally:
            knn.ENABLE_CROSS_VALIDATION = original_knn_cv

    knn_best_sig = _make_signature(
        {
            "section": "knn_best",
            "context": run_context,
            "constants": knn_consts,
        }
    )
    knn_best, changed = _get_or_compute(
        cache=cache,
        key="knn_best",
        signature=knn_best_sig,
        compute_fn=_compute_knn_best,
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    cache_changed = cache_changed or changed

    tof_sig = _make_signature(
        {
            "section": "tof_curves",
            "context": run_context,
            "dt_best": dt_best,
            "svm_best": svm_best,
            "knn_best": knn_best,
            "ann_best": ann_best,
            "tof_weight_grid": TOF_WEIGHT_GRID,
            "constants": {
                "dt": dt_consts,
                "svm": svm_consts,
                "knn": knn_consts,
                "ann": ann_consts,
            },
        }
    )
    curves, changed = _get_or_compute(
        cache=cache,
        key="tof_curves",
        signature=tof_sig,
        compute_fn=lambda: evaluate_tof_weight_curves(
            dt_best=dt_best,
            svm_best=svm_best,
            knn_best=knn_best,
            ann_best=ann_best,
            dt_ds=get_dt_dataset()[0],
            dt_feature_names=get_dt_dataset()[1],
            svm_ds=get_svm_dataset()[0],
            svm_feature_names=get_svm_dataset()[1],
            knn_ds=get_knn_dataset()[0],
            knn_feature_names=get_knn_dataset()[1],
            ann_ds=get_ann_dataset()[0],
            ann_feature_names=get_ann_dataset()[1],
            ann_device=ann_device,
            ann_verbose=ann_verbose,
            seed=seed,
        ),
        use_cache=use_cache,
        write_cache=write_cache,
        force_recompute=force_recompute,
    )
    cache_changed = cache_changed or changed

    svm_poly_plot = output_dir / "svm_poly_accuracy_contour.jpg"
    svm_linear_rbf_plot = output_dir / "svm_linear_rbf_accuracy.jpg"
    ann_plot = output_dir / "ann_accuracy_contour.jpg"
    tof_plot = output_dir / "tof_gated_weight_accuracy_comparison.jpg"

    save_contour_plot(
        x_values=SVM_POLY_DEGREE_GRID,
        y_values=SVM_C_GRID,
        z_values=poly_z,
        title="SVM Polynomial Accuracy Contour",
        xlabel="DEFAULT_POLY_DEGREE",
        ylabel="DEFAULT_C",
        out_path=svm_poly_plot,
        y_log_scale=True,
    )
    save_linear_rbf_plot(
        c_values=SVM_C_GRID,
        linear_acc=linear_acc,
        rbf_acc=rbf_acc,
        out_path=svm_linear_rbf_plot,
    )
    save_contour_plot(
        x_values=ANN_NEURONS_GRID,
        y_values=ANN_LAYERS_GRID,
        z_values=ann_z,
        title="ANN Accuracy Contour",
        xlabel="Neurons Per Hidden Layer",
        ylabel="Number of Hidden Layers",
        out_path=ann_plot,
    )

    dt_depth = dt_best.get("max_depth", None)
    dt_depth_text = "None" if dt_depth is None else str(dt_depth)
    legend_labels = {
        "svm_poly": f"SVM Poly (deg={svm_best['svm_poly']['degree']}, C={svm_best['svm_poly']['C']})",
        "svm_rbf": f"SVM RBF (C={svm_best['svm_rbf']['C']})",
        "svm_linear": f"SVM Linear (C={svm_best['svm_linear']['C']})",
        "knn": (
            "KNN "
            f"(k={knn_best['n_neighbors']}, w={knn_best['weights']}, "
            f"m={knn_best['metric']}, p={knn_best['p']}, "
            f"alg={knn_best['algorithm']}, leaf={knn_best['leaf_size']})"
        ),
        "dt": (
            "DT "
            f"(crit={dt_best.get('criterion', dt.DT_CRITERION)}, "
            f"split={dt_best['min_samples_split']}, leaf={dt_best['min_samples_leaf']}, "
            f"depth={dt_depth_text})"
        ),
        "ann": f"ANN (layers={ann_best['layers']}, neurons={ann_best['neurons_per_layer']})",
    }
    save_tof_line_plot(
        x_values=TOF_WEIGHT_GRID,
        curves={
            "svm_poly": curves["tof_svm_poly"],
            "svm_rbf": curves["tof_svm_rbf"],
            "svm_linear": curves["tof_svm_linear"],
            "knn": curves["tof_knn"],
            "dt": curves["tof_dt"],
            "ann": curves["tof_ann"],
        },
        legend_labels=legend_labels,
        out_path=tof_plot,
    )

    if write_cache and cache_changed:
        _save_cache(cache_path, cache)
        print(f"[CACHE-WRITE] updated cache saved to {cache_path}")
    elif write_cache:
        print("[CACHE-WRITE] no cache update needed.")
    else:
        print("[CACHE-WRITE] skipped by --no_cache_write.")

    print("\nSaved files:")
    print(f"1) {svm_poly_plot}")
    print(f"2) {svm_linear_rbf_plot}")
    print(f"3) {ann_plot}")
    print(f"4) {tof_plot}")


if __name__ == "__main__":
    main()
