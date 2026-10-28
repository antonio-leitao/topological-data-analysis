"""Draw one figure per scaling benchmark panel from its CSV."""

import csv
from collections import defaultdict
from math import log
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path(__file__).parent / "results"
ENGINES = ("gudhi", "tda", "tda_quotient")
LABELS = {"gudhi": "GUDHI", "tda": "TDA (exact)", "tda_quotient": "TDA (quotient)"}
COLORS = {"green": "#34a168", "blue": "#4d81d0", "orange": "#e57e39", "red": "#ed6b75"}
THEMES = {
    "light": dict(surface="#ffffff", text="#000000", muted="#666666", baseline="#a6a6a6"),
    "dark": dict(surface="#1a1a19", text="#f2f2f2", muted="#a3a3a3", baseline="#7c7c7c"),
}
SIZE = (3.8, 3)  # inches, one panel per figure
FIGURES = {"a": "scaling_sample", "b": "scaling_dim", "c": "scaling_error",
           "c_bounds": "scaling_error_bounds", "b1": "scaling_reach", "b2": "scaling_size"}
# Reference lines `scaling_error_bounds` adds to the log 3 bound.
TIGHTER = ((log(3.0) / 2, r"log $\sqrt{3}$ bound"),)
DATASETS = {"celegans": "C. elegans", "vicsek": "Vicsek", "hiv1": "HIV-1",
            "klein": "Klein bottle", "dragon": "dragon", "o3": "O(3)"}


def style(theme):
    """Settings every figure shares: no top or right spine, no grid, frameless
    legends, and markers ringed in the background colour."""
    return {
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "savefig.dpi": 200,
        "text.color": theme["text"],
        "axes.edgecolor": theme["text"],
        "axes.labelcolor": theme["text"],
        "xtick.color": theme["text"],
        "ytick.color": theme["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "lines.markeredgewidth": 2,
        "lines.markeredgecolor": theme["surface"],
        "mathtext.default": "regular",
    }


def colors(theme):
    """The baseline in grey, ours in colour; the quotient is green throughout."""
    return {"gudhi": theme["baseline"], "tda": COLORS["blue"], "tda_quotient": COLORS["green"]}


def read(path):
    """Return no rows for a panel not yet run."""
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def curve(rows, panel, engine, xkey):
    """Median seconds per size. A size where any repetition failed is left out,
    so the curve ends at the engine's last completed size."""
    times, failed = defaultdict(list), set()
    for row in rows:
        if row["panel"] != panel or row["engine"] != engine:
            continue
        if row["status"] == "ok":
            times[int(row[xkey])].append(float(row["seconds"]))
        else:
            failed.add(int(row[xkey]))
    xs = sorted(x for x in times if x not in failed)
    return xs, [median(times[x]) for x in xs]


def homology(dims):
    return [f"$H_{{{d}}}$" for d in dims]


def build_timing(rows, panel, theme, out):
    """Median time per engine: against sample size in panel a, against
    homology dimension in panel b."""
    xkey = "n" if panel == "a" else "max_dim"
    fig, ax = plt.subplots(figsize=SIZE)
    for engine, color in colors(theme).items():
        xs, ys = curve(rows, panel, engine, xkey)
        if xs:
            ax.plot(xs, ys, marker="o", color=color, label=LABELS[engine])
    ax.set_yscale("log")
    if panel == "a":
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
    else:
        dims = sorted({int(row[xkey]) for row in rows})
        ax.set_xticks(dims, homology(dims))
        ax.set_xlim(dims[0] - 0.5, dims[-1] + 0.5)
        ax.set_xlabel("Homology dimension")
    ax.set_ylabel("Time (s)")
    ax.legend()
    fig.tight_layout()
    save(fig, out)


def build_c(rows, theme, out, extra=()):
    """Bottleneck distance between exact and quotient barcodes, per dimension.

    `extra` holds more (value, label) reference lines, labelled like the log 3
    one. They sit behind the boxes, so everything else is drawn exactly as
    without them."""
    dims = sorted({int(r["homology_dim"]) for r in rows})
    data = [[float(r["bottleneck"]) for r in rows if int(r["homology_dim"]) == d] for d in dims]
    green = COLORS["green"]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.boxplot(data, positions=range(len(dims)), widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=green + "40", edgecolor=green, linewidth=1.5),
               medianprops=dict(color=theme["text"], linewidth=1.5),
               whiskerprops=dict(color=green, linewidth=1.5),
               capprops=dict(color=green, linewidth=1.5),
               flierprops=dict(marker="o", markersize=4, markerfacecolor=green,
                               markeredgecolor="none"))

    ax.axhline(log(3.0), color=theme["muted"], linewidth=1.5, linestyle="--")
    ax.text(-0.4, log(3.0) * 1.08, "log 3 bound", va="bottom", fontsize=9,
            color=theme["muted"])
    for value, text in extra:
        ax.axhline(value, color=theme["muted"], linewidth=1.5, linestyle="--", zorder=1)
        ax.text(-0.4, value * 1.08, text, va="bottom", fontsize=9, color=theme["muted"])
    ax.set_yscale("log")
    # Exact matches remain off the logarithmic axis and are counted separately.
    positive = [v for column in data for v in column if v > 0.0]
    floor = (min(positive) if positive else log(3.0) * 1e-3) * 0.55
    ax.set_ylim(floor, log(3.0) * 1.9)
    exact = sum(v <= 0.0 for column in data for v in column)
    if exact:
        ax.text(0.98, 0.03, f"{exact} exact (0, off-scale)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=theme["muted"])
    ax.set_xticks(range(len(dims)), homology(dims))
    ax.set_xlabel("Homology dimension")
    ax.set_ylabel("Bottleneck distance")
    fig.tight_layout()
    save(fig, out)


def compact(n):
    """In K and M, one decimal below 100: 2,500 -> 2.5K, 546,815,948 -> 547M."""
    for scale, unit in ((1e6, "M"), (1e3, "K")):
        if n >= scale:
            value = n / scale
            return (f"{value:.1f}" if value < 100 else f"{value:.0f}").removesuffix(".0") + unit
    return f"{n:,}"


def reach(rows):
    """Per dimension: the largest n finished and what stopped the sweep:
    timeout, oom, error, or cap when it ran out of sizes."""
    out = {}
    for dim in sorted({int(r["dim"]) for r in rows}):
        mine = [r for r in rows if int(r["dim"]) == dim]
        done = [int(r["n"]) for r in mine if r["status"] == "ok"]
        if done:
            stop = next((r["status"] for r in mine if r["status"] != "ok"), "cap")
            out[dim] = (max(done), stop)
    return out


def build_b1(rows, theme, out):
    """Largest sample the quotient finishes per homology dimension."""
    found = reach(rows)
    dims = sorted(found)
    ns = [found[d][0] for d in dims]
    fig, ax = plt.subplots(figsize=SIZE)

    ax.plot(dims, ns, marker="o", color=COLORS["green"])
    for dim, n in zip(dims, ns):
        text = ("≥ " if found[dim][1] == "cap" else "") + compact(n)
        ax.annotate(text, (dim, n), textcoords="offset points", xytext=(0, 8), ha="center",
                    va="bottom", fontsize=9, color=theme["muted"])

    ax.set_yscale("log")
    ax.set_ylim(min(ns) / 2, max(ns) * 4)  # room for the labels above the points
    ax.set_xticks(dims, homology(dims))
    ax.set_xlim(dims[0] - 0.5, dims[-1] + 0.5)
    ax.set_xlabel("Homology dimension")
    ax.set_ylabel("Sample size")
    fig.tight_layout()
    save(fig, out)


def build_b2(rows, theme, out):
    """Filtration size without and with the quotient, one row per dataset."""
    sizes, points = defaultdict(dict), {}
    for row in rows:
        if row["status"] == "ok":
            sizes[row["dataset"]][row["pipeline"]] = int(row["simplices"])
            points[row["dataset"]] = int(row["n"])
    names = sorted(sizes, key=lambda name: max(sizes[name].values()))  # largest on top
    grey, green = theme["baseline"], COLORS["green"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for y, name in enumerate(names):
        s = sizes[name]
        tag = dict(textcoords="offset points", va="center", fontsize=9, color=theme["muted"])
        if "vanilla" in s and "quotient" in s:
            ax.plot([s["quotient"], s["vanilla"]], [y, y], color=grey, zorder=1)
        if "vanilla" in s:
            ax.plot([s["vanilla"]], [y], marker="o", color=grey, zorder=2)
            ax.annotate(compact(s["vanilla"]), (s["vanilla"], y), xytext=(8, 0), ha="left",
                        **tag)
        if "quotient" in s:
            ax.plot([s["quotient"]], [y], marker="o", color=green, zorder=3)
            ax.annotate(compact(s["quotient"]), (s["quotient"], y), xytext=(-8, 0),
                        ha="right", **tag)
    every = [v for s in sizes.values() for v in s.values()]
    ax.set_xscale("log")
    ax.set_xlim(min(every) / 6, max(every) * 6)  # room for the end labels
    ax.set_yticks(range(len(names)), [f"{label(name)} ({points[name]:,})" for name in names])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Simplices up to dimension {rows[0]['max_dim']}")
    # Below the axis: every row reaches across the plot, so no corner is free.
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=grey, label="Without quotient"),
        Line2D([], [], marker="o", linestyle="none", color=green, label="With quotient"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out)


def label(dataset):
    """Display name: `klein_400` reads as Klein bottle; the size is shown apart."""
    return DATASETS.get(dataset.split("_")[0], dataset)


def save(fig, out):
    fig.savefig(f"{out}.png")
    # No creation date, so redrawing the same results leaves the PDF unchanged.
    fig.savefig(f"{out}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out}.png and {out}.pdf")


def main():
    panels = {p: read(RESULTS / f"{p}.csv") for p in ("a", "b", "c", "b1", "b2")}
    for panel, rows in panels.items():
        print(f"panel {panel}: {len(rows)} rows" + ("" if rows else "  (not run yet)"))
    if not any(panels.values()):
        raise SystemExit(f"no results in {RESULTS}; run scaling.py first")

    # Panel c records a failed (dim, seed) with no distance.
    c_rows = [r for r in panels["c"] if r.get("status", "ok") == "ok" and r["homology_dim"]]
    for name, suffix in (("light", ""), ("dark", "_dark")):
        theme = THEMES[name]
        out = {panel: RESULTS / f"{figure}{suffix}" for panel, figure in FIGURES.items()}
        with plt.rc_context(style(theme)):
            for panel in ("a", "b"):
                if any(row["status"] == "ok" for row in panels[panel]):
                    build_timing(panels[panel], panel, theme, out[panel])
            if c_rows:
                build_c(c_rows, theme, out["c"])
                build_c(c_rows, theme, out["c_bounds"], extra=TIGHTER)
            if reach(panels["b1"]):
                build_b1(panels["b1"], theme, out["b1"])
            if any(row["status"] == "ok" for row in panels["b2"]):
                build_b2(panels["b2"], theme, out["b2"])


if __name__ == "__main__":
    main()
