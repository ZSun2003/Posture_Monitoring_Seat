#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    labels = [
        "straight",
        "forward",
        "back",
        "middle",
        "left-cross",
        "right-cross",
        "lying down",
        "left",
        "right",
        "sit-at-front",
    ]

    # Rows: real classes, Columns: predicted classes
    cm = np.array(
        [
            [134, 0, 0, 16, 0, 0, 0, 0, 0, 0],   # straight
            [0, 150, 0, 0, 0, 0, 0, 0, 0, 0],    # forward
            [0, 0, 150, 0, 0, 0, 0, 0, 0, 0],    # back
            [9, 0, 0, 141, 0, 0, 0, 0, 0, 0],    # middle
            [0, 0, 0, 0, 150, 0, 0, 0, 0, 0],    # left-cross
            [0, 0, 0, 0, 0, 116, 0, 34, 0, 0],   # right-cross
            [0, 0, 0, 0, 0, 0, 150, 0, 0, 0],    # lying down
            [0, 0, 0, 0, 0, 0, 0, 150, 0, 0],    # left
            [0, 0, 0, 0, 0, 0, 0, 0, 150, 0],    # right
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 150],    # sit-at-front
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("real")
    ax.set_title("Confusion Matrix")

    max_val = int(cm.max())
    threshold = max_val * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val == 0:
                continue
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=11)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "confusion_matrix_from_table.jpg"
    fig.savefig(out_path, format="jpeg", dpi=300)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
