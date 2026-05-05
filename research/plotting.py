"""Plotting helpers for research notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_default_style() -> None:
    """Apply a readable default Matplotlib style for static notebook output."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": True,
        }
    )
