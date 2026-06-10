"""Plotting setup for manuscript figures."""

from __future__ import annotations


def apply_manuscript_style() -> None:
    """Apply a conservative Matplotlib style used by the study scripts."""
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["DejaVu Serif", "Times", "serif"]
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titleweight"] = "bold"
