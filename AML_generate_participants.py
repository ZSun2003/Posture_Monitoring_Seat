#!/usr/bin/env python3
"""
Generate synthetic AML participants by mixing multiple source participants, then perturbing.

Default behavior:
    Mix participant_1/2/3 in each posture, then generate participant_5/6/7.

Expected dataset layout:
    AML_DATASET/
      Straight/participant_1.csv
      Left/participant_2.csv
      ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


FSR_NORM_COLS: List[str] = [f"fsr{i}_norm" for i in range(10)]
FSR_STD_COLS: List[str] = [f"fsr{i}_std" for i in range(10)]
RATIO_COLS: List[str] = ["lr_ratio", "ud_ratio", "cop_x_ratio", "cop_y_ratio"]
TOF_COLS: List[str] = ["tof_norm", "tof_gated"]
TOF_USE_COL = "tof_use"

DEFAULT_SOURCE_IDS = "1,2,3"
DEFAULT_TARGET_IDS = "5,6,7"
REQUIRED_COLS = FSR_NORM_COLS + RATIO_COLS + TOF_COLS + [TOF_USE_COL]


@dataclass
class PerturbConfig:
    fsr_scale_low: float = 0.92
    fsr_scale_high: float = 1.08
    fsr_bias_std: float = 0.003
    fsr_noise_std: float = 0.006
    fsr_std_scale_low: float = 0.90
    fsr_std_scale_high: float = 1.10
    fsr_std_noise_std: float = 0.10
    ratio_scale_low: float = 0.96
    ratio_scale_high: float = 1.04
    ratio_noise_std: float = 0.01
    tof_scale_low: float = 0.95
    tof_scale_high: float = 1.05
    tof_noise_std: float = 0.02
    tof_flip_prob: float = 0.01
    round_digits: int = 4


@dataclass
class ParticipantProfile:
    fsr_scale: np.ndarray
    fsr_bias: np.ndarray
    fsr_std_scale: np.ndarray
    ratio_scale: np.ndarray
    ratio_bias: np.ndarray
    tof_scale: np.ndarray
    tof_bias: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate new participant files by mixing source participants and then perturbing."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("AML_DATASET"),
        help="Dataset root folder containing posture subfolders.",
    )
    parser.add_argument(
        "--source-ids",
        type=str,
        default=DEFAULT_SOURCE_IDS,
        help="Comma-separated source participant ids, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--target-ids",
        type=str,
        default=DEFAULT_TARGET_IDS,
        help="Comma-separated target participant ids, e.g. '5,6,7'.",
    )
    parser.add_argument(
        "--rows-per-output",
        type=int,
        default=0,
        help="Rows per generated target file. 0 means min(rows among source files in that posture).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files if they exist.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be generated.")

    parser.add_argument("--fsr-scale-low", type=float, default=0.92, help="Lower bound of FSR multiplicative scale.")
    parser.add_argument("--fsr-scale-high", type=float, default=1.08, help="Upper bound of FSR multiplicative scale.")
    parser.add_argument("--fsr-bias-std", type=float, default=0.003, help="Std of per-participant FSR bias.")
    parser.add_argument("--fsr-noise-std", type=float, default=0.006, help="Std of per-sample FSR additive noise.")
    parser.add_argument("--ratio-noise-std", type=float, default=0.01, help="Std of additive noise for ratio features.")
    parser.add_argument("--tof-noise-std", type=float, default=0.02, help="Std of additive noise for ToF features.")
    parser.add_argument("--tof-flip-prob", type=float, default=0.01, help="Probability to flip tof_use (0<->1).")
    parser.add_argument(
        "--round-digits", type=int, default=4, help="Round float columns to N digits before writing CSV."
    )
    return parser.parse_args()


def parse_participant_ids(ids_text: str, arg_name: str) -> List[int]:
    ids: List[int] = []
    seen: set[int] = set()
    for chunk in ids_text.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            pid = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid id '{item}' in --{arg_name}.") from exc
        if pid <= 0:
            raise ValueError(f"Participant id must be >=1 in --{arg_name}, got {pid}.")
        if pid in seen:
            raise ValueError(f"Duplicate participant_{pid} in --{arg_name}.")
        seen.add(pid)
        ids.append(pid)
    if not ids:
        raise ValueError(f"--{arg_name} has no valid participant ids.")
    return ids


def discover_posture_dirs(data_root: Path) -> List[Path]:
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    postures = [p for p in data_root.iterdir() if p.is_dir() and any(p.glob("participant_*.csv"))]
    postures.sort(key=lambda p: p.name.lower())
    if not postures:
        raise RuntimeError(f"No posture folders with participant_*.csv found under: {data_root}")
    return postures


def validate_source_files(postures: Sequence[Path], source_ids: Sequence[int]) -> None:
    for posture_dir in postures:
        for source_id in source_ids:
            src = posture_dir / f"participant_{source_id}.csv"
            if not src.exists():
                raise FileNotFoundError(f"Missing source file: {src}")


def stable_name_seed(name: str) -> int:
    value = 0
    for ch in name:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return value


def clip_column(values: np.ndarray, column: str) -> np.ndarray:
    if column.startswith("fsr") and column.endswith("_norm"):
        return np.clip(values, 0.0, 1.0)
    if column == "lr_ratio":
        return np.clip(values, 0.0, 3.0)
    if column == "ud_ratio":
        return np.clip(values, 0.0, 1.5)
    if column in ("cop_x_ratio", "cop_y_ratio"):
        return np.clip(values, 0.0, 1.0)
    if column in ("tof_norm", "tof_gated"):
        return np.clip(values, 0.0, 1.0)
    return values


def require_columns(df: pd.DataFrame, file_path: Path, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {missing}")


def load_source_frames(posture_dir: Path, source_ids: Sequence[int]) -> Dict[int, pd.DataFrame]:
    frames: Dict[int, pd.DataFrame] = {}
    for source_id in source_ids:
        src_path = posture_dir / f"participant_{source_id}.csv"
        df = pd.read_csv(src_path, encoding="utf-8-sig")
        require_columns(df, src_path, REQUIRED_COLS)
        frames[int(source_id)] = df

    template_cols = list(frames[source_ids[0]].columns)
    template_set = set(template_cols)
    for source_id in source_ids:
        df = frames[source_id]
        if set(df.columns) != template_set:
            raise ValueError(
                f"Column mismatch in {posture_dir}/participant_{source_id}.csv, "
                "source participants must have consistent columns."
            )
        frames[source_id] = df[template_cols]
    return frames


def resolve_output_rows(source_frames: Dict[int, pd.DataFrame], rows_per_output: int) -> int:
    if rows_per_output > 0:
        return int(rows_per_output)
    source_lengths = [len(df) for df in source_frames.values()]
    return int(min(source_lengths))


def mix_source_frames(
    source_frames: Dict[int, pd.DataFrame],
    source_ids: Sequence[int],
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")

    n_sources = len(source_ids)
    base = n_rows // n_sources
    remainder = n_rows - base * n_sources
    quotas = np.full(n_sources, base, dtype=int)
    if remainder > 0:
        extra_idx = rng.choice(n_sources, size=remainder, replace=False)
        quotas[extra_idx] += 1

    parts: List[pd.DataFrame] = []
    for i, source_id in enumerate(source_ids):
        df = source_frames[source_id]
        take = int(quotas[i])
        if take <= 0:
            continue
        replace = take > len(df)
        sampled_idx = rng.choice(len(df), size=take, replace=replace)
        parts.append(df.iloc[sampled_idx].copy())

    mixed = pd.concat(parts, ignore_index=True)
    mixed = mixed.iloc[rng.permutation(len(mixed))].reset_index(drop=True)

    if "t_ms" in mixed.columns:
        base_time = source_frames[source_ids[0]]["t_ms"].to_numpy()
        if len(base_time) >= n_rows:
            mixed.loc[:, "t_ms"] = base_time[:n_rows]
        else:
            step = np.median(np.diff(base_time)) if len(base_time) >= 2 else 100
            step = float(step) if np.isfinite(step) and step > 0 else 100.0
            start = float(base_time[0]) if len(base_time) else 0.0
            mixed.loc[:, "t_ms"] = start + np.arange(n_rows) * step

    return mixed


def sample_profile(rng: np.random.Generator, cfg: PerturbConfig) -> ParticipantProfile:
    fsr_scale = rng.uniform(cfg.fsr_scale_low, cfg.fsr_scale_high, size=len(FSR_NORM_COLS))
    fsr_bias = rng.normal(0.0, cfg.fsr_bias_std, size=len(FSR_NORM_COLS))
    fsr_std_scale = rng.uniform(cfg.fsr_std_scale_low, cfg.fsr_std_scale_high, size=len(FSR_STD_COLS))
    ratio_scale = rng.uniform(cfg.ratio_scale_low, cfg.ratio_scale_high, size=len(RATIO_COLS))
    ratio_bias = rng.normal(0.0, cfg.ratio_noise_std, size=len(RATIO_COLS))
    tof_scale = rng.uniform(cfg.tof_scale_low, cfg.tof_scale_high, size=len(TOF_COLS))
    tof_bias = rng.normal(0.0, cfg.tof_noise_std, size=len(TOF_COLS))
    return ParticipantProfile(
        fsr_scale=fsr_scale.astype(np.float64),
        fsr_bias=fsr_bias.astype(np.float64),
        fsr_std_scale=fsr_std_scale.astype(np.float64),
        ratio_scale=ratio_scale.astype(np.float64),
        ratio_bias=ratio_bias.astype(np.float64),
        tof_scale=tof_scale.astype(np.float64),
        tof_bias=tof_bias.astype(np.float64),
    )


def perturb_dataframe(
    df: pd.DataFrame,
    profile: ParticipantProfile,
    cfg: PerturbConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, Path("<dataframe>"), REQUIRED_COLS)

    fsr = out[FSR_NORM_COLS].to_numpy(dtype=np.float64)
    fsr = fsr * profile.fsr_scale + profile.fsr_bias
    fsr += rng.normal(0.0, cfg.fsr_noise_std, size=fsr.shape)
    fsr = np.clip(fsr, 0.0, 1.0)
    out.loc[:, FSR_NORM_COLS] = fsr

    std_cols = [c for c in FSR_STD_COLS if c in out.columns]
    if std_cols:
        idx = np.array([FSR_STD_COLS.index(c) for c in std_cols], dtype=int)
        fsr_std = out[std_cols].to_numpy(dtype=np.float64)
        fsr_std = fsr_std * profile.fsr_std_scale[idx]
        fsr_std += rng.normal(0.0, cfg.fsr_std_noise_std, size=fsr_std.shape)
        fsr_std = np.clip(fsr_std, 0.0, None)
        out.loc[:, std_cols] = fsr_std

    ratio = out[RATIO_COLS].to_numpy(dtype=np.float64)
    ratio = ratio * profile.ratio_scale + profile.ratio_bias
    ratio += rng.normal(0.0, cfg.ratio_noise_std, size=ratio.shape)
    for i, c in enumerate(RATIO_COLS):
        ratio[:, i] = clip_column(ratio[:, i], c)
    out.loc[:, RATIO_COLS] = ratio

    tof = out[TOF_COLS].to_numpy(dtype=np.float64)
    tof = tof * profile.tof_scale + profile.tof_bias
    tof += rng.normal(0.0, cfg.tof_noise_std, size=tof.shape)
    for i, c in enumerate(TOF_COLS):
        tof[:, i] = clip_column(tof[:, i], c)
    out.loc[:, TOF_COLS] = tof

    tof_use = (out[TOF_USE_COL].to_numpy(dtype=np.float64) > 0.5).astype(np.int8)
    if cfg.tof_flip_prob > 0.0:
        prob = float(np.clip(cfg.tof_flip_prob, 0.0, 1.0))
        flip = rng.random(size=tof_use.shape[0]) < prob
        tof_use[flip] = 1 - tof_use[flip]
    out.loc[:, TOF_USE_COL] = tof_use
    out.loc[tof_use == 0, "tof_gated"] = 0.5

    if "t_ms" in out.columns:
        out.loc[:, "t_ms"] = df["t_ms"].to_numpy()

    if cfg.round_digits >= 0:
        float_cols = [
            col for col in out.columns
            if col != TOF_USE_COL and pd.api.types.is_numeric_dtype(out[col])
        ]
        out.loc[:, float_cols] = out[float_cols].round(cfg.round_digits)

    out.loc[:, TOF_USE_COL] = out[TOF_USE_COL].astype(int)
    return out


def build_cfg(args: argparse.Namespace) -> PerturbConfig:
    low, high = sorted((args.fsr_scale_low, args.fsr_scale_high))
    return PerturbConfig(
        fsr_scale_low=low,
        fsr_scale_high=high,
        fsr_bias_std=float(max(args.fsr_bias_std, 0.0)),
        fsr_noise_std=float(max(args.fsr_noise_std, 0.0)),
        ratio_noise_std=float(max(args.ratio_noise_std, 0.0)),
        tof_noise_std=float(max(args.tof_noise_std, 0.0)),
        tof_flip_prob=float(np.clip(args.tof_flip_prob, 0.0, 1.0)),
        round_digits=int(args.round_digits),
    )


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    source_ids = parse_participant_ids(args.source_ids, "source-ids")
    target_ids = parse_participant_ids(args.target_ids, "target-ids")
    overlap = set(source_ids) & set(target_ids)
    if overlap:
        raise ValueError(
            f"source ids and target ids overlap: {sorted(overlap)}. "
            "Use different ids to avoid overwriting source participants."
        )

    data_root = args.data_root.expanduser().resolve()
    postures = discover_posture_dirs(data_root)
    validate_source_files(postures, source_ids)

    profile_rng = np.random.default_rng(args.seed)
    target_profiles: Dict[int, ParticipantProfile] = {}
    for target_id in target_ids:
        target_profiles[target_id] = sample_profile(profile_rng, cfg)

    written = 0
    skipped = 0

    print(f"Data root: {data_root}")
    print(f"Postures ({len(postures)}): {[p.name for p in postures]}")
    print(f"Source ids: {source_ids}")
    print(f"Target ids: {target_ids}")
    print(f"Rows per output: {args.rows_per_output} (0 means min source rows per posture)")
    print(f"Seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")

    for posture_dir in postures:
        source_frames = load_source_frames(posture_dir, source_ids)
        n_rows = resolve_output_rows(source_frames, int(args.rows_per_output))

        for target_id in target_ids:
            dst_path = posture_dir / f"participant_{target_id}.csv"

            if dst_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[SKIP] {dst_path} already exists (use --overwrite to replace).")
                continue

            seed_mix = (
                (int(args.seed) & 0xFFFFFFFF)
                ^ ((int(target_id) * 2654435761) & 0xFFFFFFFF)
                ^ stable_name_seed(posture_dir.name)
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(seed_mix)

            mixed_df = mix_source_frames(source_frames, source_ids, n_rows, rng)
            out_df = perturb_dataframe(mixed_df, target_profiles[target_id], cfg, rng)
            if args.dry_run:
                print(
                    f"[DRY ] participant_{target_id}.csv <= mix{source_ids} "
                    f"({posture_dir.name}, rows={len(out_df)})"
                )
                continue

            out_df.to_csv(
                dst_path,
                index=False,
                encoding="utf-8-sig",
                float_format=f"%.{cfg.round_digits}f",
            )
            written += 1
            print(
                f"[WRITE] participant_{target_id}.csv <= mix{source_ids} "
                f"({posture_dir.name}, rows={len(out_df)})"
            )

    print(f"Done. written={written}, skipped={skipped}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
