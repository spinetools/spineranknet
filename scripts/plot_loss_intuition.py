"""Generate presentation figures illustrating the SpineRank loss.

Figures use the ETH Zürich corporate palette and Helvetica (sans-serif math).
Produces, under ``results/figures/`` (override with ``--out``):

* ``hinge_margin_<m>.pdf``   one squared-hinge curve per margin
* ``hinge_margins.pdf``      all margins overlaid on one axes
* ``sim_parabola.pdf``       the same-grade similarity penalty
* ``hinge_margin_sweep.gif`` animation sweeping the adaptive margin

The squared hinge penalises an ordinal pair (i, j) of *different* grades:

    L_hinge(x) = max(0, m - x)^2,   x = y_ij * (s_i - s_j)

where the margin ``m = margin_base + margin_scale * |g_i - g_j| / (K-1) * ...``
grows with how far apart the two grades are (0.5 for neighbours, up to 2.0 for
the extremes). The similarity term penalises same-grade pairs by (s_i - s_j)^2.

Usage
-----
    python scripts/plot_loss_intuition.py
    python scripts/plot_loss_intuition.py --out paper/figures --margins 0.5 1.0 1.5 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ETH Zürich corporate palette (100% tints).
ETH_BLUE = "#215CAF"
ETH_PETROL = "#007894"
ETH_GREEN = "#627313"
ETH_BRONZE = "#8E6713"
ETH_RED = "#B7352D"
ETH_PURPLE = "#A30774"
ETH_GRAY = "#6F6F6F"

# Helvetica body font with sans-serif mathtext (Computer-Modern-sans-like) so
# math labels match. Falls back to Arial / DejaVu Sans if Helvetica is absent.
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stixsans",
})

XLO, XHI, PMAX = -1.0, 4.0, 6.0
HINGE_COLOR = ETH_RED      # penalty curve / "fined"
ZONE_COLOR = ETH_RED       # shaded penalised region
MARGIN_COLOR = ETH_BLUE    # margin line / schedule line
SIM_COLOR = ETH_GREEN      # similarity parabola
CLEAR_COLOR = ETH_PETROL   # "clears margin -> zero"


def _eth_blue_shades(n: int) -> list[tuple[float, float, float]]:
    """n tints of ETH blue, light (small margin) -> full (large margin)."""
    base = np.array([0x21, 0x5C, 0xAF]) / 255.0
    shades = []
    for k in range(n):
        f = 0.30 + 0.70 * (k / max(n - 1, 1))   # mix fraction with white
        shades.append(tuple(base * f + (1.0 - f)))
    return shades


def _margin_label(m: float) -> str:
    if m <= 0.6:
        return "neighbouring grades"
    if m >= 1.9:
        return "extreme grades (0 vs 4)"
    return "intermediate distance"


def _style_hinge_axes(ax: plt.Axes) -> None:
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, PMAX)
    ax.set_xlabel(r"correct-direction gap  $y_{ij}\,(s_i-s_j)$")
    ax.set_ylabel("penalty")


def fig_single(m: float, out: Path) -> Path:
    """One squared-hinge curve for a single margin, with the penalised zone."""
    x = np.linspace(XLO, XHI, 400)
    y = np.maximum(0.0, m - x) ** 2

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axvspan(XLO, m, color=ZONE_COLOR, alpha=0.10)
    ax.plot(x, y, lw=2.6, color=HINGE_COLOR)
    ax.axvline(m, ls="--", lw=1.3, color=MARGIN_COLOR)
    ax.text(m, PMAX * 0.96, f"  margin = {m:.1f}", color=MARGIN_COLOR,
            ha="left", va="top", fontsize=11)
    ax.text((XLO + m) / 2, PMAX * 0.55, "wrong /\ntoo close\n-> fined",
            ha="center", va="center", color=ETH_RED, fontsize=10)
    ax.text((m + XHI) / 2, PMAX * 0.10, "clears margin -> zero",
            ha="center", va="center", color=CLEAR_COLOR, fontsize=10)
    ax.set_title(rf"$\mathcal{{L}}_{{\mathrm{{hinge}}}}$ — {_margin_label(m)}")
    _style_hinge_axes(ax)
    fig.tight_layout()

    path = out / f"hinge_margin_{str(m).replace('.', 'p')}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_overlay(margins: list[float], out: Path) -> Path:
    """All margins on one axes (the slide-ready comparison)."""
    x = np.linspace(XLO, XHI, 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    shades = _eth_blue_shades(len(margins))
    for k, m in enumerate(margins):
        c = shades[k]
        ax.plot(x, np.maximum(0.0, m - x) ** 2, lw=2.4, color=c,
                label=fr"$m={m:.1f}$  ({_margin_label(m)})")
        ax.axvline(m, ls=":", lw=0.9, color=c, alpha=0.7)
    ax.set_title(r"$\mathcal{L}_{\mathrm{hinge}}$: bigger grade gap $\Rightarrow$ wider margin")
    _style_hinge_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    path = out / "hinge_margins.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_sim(out: Path) -> Path:
    """Similarity (same-grade) parabola."""
    d = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(d, d ** 2, lw=2.6, color=SIM_COLOR)
    ax.set_title(r"$\mathcal{L}_{\mathrm{sim}}$ — same grade should agree")
    ax.set_xlabel(r"$s_i - s_j$  (two same-grade discs)")
    ax.set_ylabel("penalty")
    ax.set_ylim(0, PMAX)
    fig.tight_layout()
    path = out / "sim_parabola.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_margin_schedule(out: Path, margin_base: float = 0.5,
                        margin_scale: float = 1.5, K: int = 5) -> Path:
    """How the margin grows with grade distance: m = base + scale * w_norm."""
    w = np.linspace(0.0, 1.0, 200)            # normalised grade distance in [0,1]
    m = margin_base + margin_scale * w

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(w, m, lw=2.6, color=MARGIN_COLOR)

    # intercept (margin_base) and span (margin_scale)
    ax.axhline(margin_base, ls=":", lw=1, color=ETH_GRAY)
    ax.annotate(rf"$m_0 = {margin_base:.1f}$  (base, neighbours)",
                xy=(0, margin_base), xytext=(0.30, margin_base - 0.28),
                fontsize=10, color=ETH_GRAY)
    ax.annotate("", xy=(1.02, margin_base + margin_scale), xytext=(1.02, margin_base),
                arrowprops=dict(arrowstyle="<->", color=ETH_RED, lw=1.2))
    ax.text(1.05, margin_base + margin_scale / 2,
            rf"$m_s = {margin_scale:.1f}$" + "\n(scale)",
            fontsize=10, color=ETH_RED, va="center")

    # example grade-distance ticks for a K-grade task
    deltas = np.arange(K)                      # |g_i - g_j| = 0..K-1
    wd = deltas / (K - 1)
    md = margin_base + margin_scale * wd
    ax.scatter(wd, md, color=HINGE_COLOR, zorder=5, s=36)
    for d, wi, mi in zip(deltas, wd, md):
        ax.annotate(rf"$\Delta g={d}$", (wi, mi), textcoords="offset points",
                    xytext=(-2, -14), fontsize=9, color=ETH_RED, ha="center")

    ax.set_title(r"adaptive margin:  $m = m_0 + m_s\,\dfrac{|g_i-g_j|}{K-1}$")
    ax.set_xlabel(r"grade distance  $|g_i-g_j|/(K-1)$")
    ax.set_ylabel("required margin  $m$")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, margin_base + margin_scale + 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    path = out / "margin_schedule.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def anim_sweep(out: Path, fps: int = 20) -> Path:
    """Animate the margin sweeping 0.5 -> 2.0 -> 0.5 (mirrors the widget)."""
    x = np.linspace(XLO, XHI, 400)
    up = np.linspace(0.5, 2.0, 45)
    margins = np.concatenate([up, up[::-1]])

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    _style_hinge_axes(ax)
    zone = ax.axvspan(XLO, 0.5, color=ZONE_COLOR, alpha=0.10)
    (line,) = ax.plot(x, np.maximum(0.0, 0.5 - x) ** 2, lw=2.6, color=HINGE_COLOR)
    vline = ax.axvline(0.5, ls="--", lw=1.3, color=MARGIN_COLOR)
    mtxt = ax.text(0.5, PMAX * 0.96, "", color=MARGIN_COLOR, ha="left", va="top",
                   fontsize=11)
    title = ax.set_title("")
    fig.tight_layout()

    def update(m: float):
        line.set_ydata(np.maximum(0.0, m - x) ** 2)
        vline.set_xdata([m, m])
        xy = zone.get_xy()
        xy[2:4, 0] = m            # right edge of the shaded span
        zone.set_xy(xy)
        mtxt.set_position((m, PMAX * 0.96))
        mtxt.set_text(f"  margin = {m:.1f}")
        title.set_text(rf"$\mathcal{{L}}_{{\mathrm{{hinge}}}}$ — {_margin_label(m)}")
        return line, vline, zone, mtxt, title

    ani = FuncAnimation(fig, update, frames=margins, blit=False, interval=1000 / fps)
    path = out / "hinge_margin_sweep.gif"
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--margins", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--no-gif", action="store_true", help="Skip the (slow) GIF render")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    made = [fig_single(m, args.out) for m in args.margins]
    made.append(fig_overlay(args.margins, args.out))
    made.append(fig_margin_schedule(args.out))
    made.append(fig_sim(args.out))
    if not args.no_gif:
        made.append(anim_sweep(args.out))

    print("Wrote:")
    for f in made:
        print(f"  {f}")


if __name__ == "__main__":
    main()
