"""Draw Figure 2, and its b1 and b2 panels, from benchmark CSVs."""

import csv
from collections import defaultdict
from math import log
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

plt.rcParams["mathtext.default"] = "regular"

RESULTS = Path(__file__).parent / "results"
ENGINES = ("gudhi", "tda", "tda_quotient")
LABELS = {"gudhi": "GUDHI", "tda": "TDA (exact)", "tda_quotient": "TDA (quotient)"}

# Categorical slots 1-3, one per engine in ENGINES order.
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#898781", grid="#e1e0d9",
                  plain="#9a988f", series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#898781", grid="#2c2c2a",
                 plain="#8a887f", series=("#3987e5", "#d95926", "#199e70")),
}
QUOTIENT = ENGINES.index("tda_quotient")
DATASETS = {"celegans": "C. elegans", "vicsek": "Vicsek", "hiv1": "HIV-1",
            "klein": "Klein bottle", "dragon": "dragon"}


def read(path):
    """Return no rows for a panel not yet run."""
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def symbol(row):
    """T for torus, S for sphere. Older CSVs have no manifold column."""
    return {"sphere": "S", "torus": "T"}.get(row.get("manifold", "torus"), "T")


def superscript(value):
    return str(value).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def count(value):
    value = int(value)
    return f"{value // 1000}k" if value >= 1000 and not value % 1000 else f"{value:,}"


def titles(a_rows, b_rows, c_rows):
    """Derive titles from recorded conditions."""
    a, b = "a · time vs sample size", "b · time vs dimension"
    if a_rows:
        shape = ("Stanford dragon" if a_rows[0].get("manifold") == "dragon"
                 else f"{symbol(a_rows[0])}{superscript(a_rows[0]['dim'])}")
        a += f"   ({shape}, max_dim {a_rows[0]['max_dim']})"
    if b_rows:
        # The manifold grows with the axis here, so name it generically.
        letter = symbol(b_rows[0])
        b += f"   ({letter}ᵈ at max_dim d, n = {count(b_rows[0]['n'])})"
    c = "c · quotient error vs exact"
    if c_rows:
        dims = sorted({int(row["manifold_dim"]) for row in c_rows})
        shown = [f"T{superscript(d)}" for d in dims]
        c += "   (" + (", … , ".join([shown[0], shown[-1]]) if len(shown) > 2
                       else ", ".join(shown)) + ")"
    return a, b, c


def placeholder(ax, theme, title, xlabel, ylabel):
    ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=theme["muted"])
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax, theme, title, xlabel, ylabel)


def curve(rows, panel, engine, xkey):
    times, failure = defaultdict(list), {}
    for row in rows:
        if row["panel"] != panel or row["engine"] != engine:
            continue
        x = int(row[xkey])
        if row["status"] == "ok":
            times[x].append(float(row["seconds"]))
        else:
            failure.setdefault(x, row["status"])
    xs = sorted(times)
    first = min(failure) if failure else None
    return xs, [median(times[x]) for x in xs], (first, failure[first]) if first else None


def timing_panel(ax, rows, panel, xkey, theme, title, xlabel, logx):
    if not any(row["panel"] == panel for row in rows):
        return placeholder(ax, theme, title, xlabel, "seconds")

    failures = []
    for engine, color in zip(ENGINES, theme["series"]):
        xs, ys, failure = curve(rows, panel, engine, xkey)
        if xs:
            ax.plot(xs, ys, color=color, marker="o", markersize=4, linewidth=2,
                    label=LABELS[engine], zorder=3)
        if failure:
            failures.append((failure[0], color, failure[1]))

    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 8)

    if failures:
        ax.axhspan(high * 1.6, high * 6.5, color=theme["text"], alpha=0.04, zorder=0)
        ax.text(0.015, 0.965, "did not complete", transform=ax.transAxes,
                va="top", fontsize=8, color=theme["muted"])
        for i, (x, color, status) in enumerate(sorted(failures)):
            y = high * (2.4 if i % 2 == 0 else 4.4)
            ax.plot([x], [y], marker="x", markersize=7, markeredgewidth=2, color=color, zorder=4)
            ax.annotate(status, (x, y), textcoords="offset points", xytext=(8, -3),
                        fontsize=7, color=theme["muted"], zorder=4)

    style_axes(ax, theme, title, xlabel, "seconds")


def error_panel(ax, rows, theme, title):
    # Rows carry a status; a failed (dim, seed) is recorded with no distance.
    rows = [r for r in rows if r.get("status", "ok") == "ok" and r["homology_dim"]]
    if not rows:
        return placeholder(ax, theme, title, "homology dimension", "bottleneck distance")

    dims = sorted({int(r["homology_dim"]) for r in rows})
    data = [[float(r["bottleneck"]) for r in rows if int(r["homology_dim"]) == d] for d in dims]
    color = theme["series"][QUOTIENT]

    box = ax.boxplot(data, positions=range(len(dims)), widths=0.55, patch_artist=True,
                     medianprops=dict(color=theme["text"], linewidth=1.5),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                                     markeredgecolor="none", alpha=0.6))
    for patch in box["boxes"]:
        patch.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.5)
    for part in ("whiskers", "caps"):
        for line in box[part]:
            line.set(color=color, linewidth=1.2)

    ax.axhline(log(3.0), color=theme["muted"], linewidth=1.5, zorder=2)
    ax.text(-0.42, log(3.0) * 0.86, "log 3 bound", ha="left", va="top",
            fontsize=8, color=theme["muted"])
    ax.set_yscale("log")
    # Exact matches remain off the logarithmic axis and are counted separately.
    positive = [v for column in data for v in column if v > 0.0]
    floor = (min(positive) if positive else log(3.0) * 1e-3) * 0.55
    ax.set_ylim(floor, log(3.0) * 1.9)
    exact = sum(v <= 0.0 for column in data for v in column)
    if exact:
        ax.text(0.985, 0.03, f"{exact} exact (0, off-scale)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=theme["muted"])
    ax.set_xticks(range(len(dims)), [f"H{d}" for d in dims])
    style_axes(ax, theme, title, "homology dimension", "bottleneck distance")


def style_axes(ax, theme, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=theme["text"], loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=theme["muted"])
    ax.set_ylabel(ylabel, fontsize=9, color=theme["muted"])
    ax.set_facecolor(theme["surface"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(theme["grid"])


def compact(n):
    """In K and M, one decimal below 100: 2,500 -> 2.5K, 546,815,948 -> 547M."""
    for scale, unit in ((1e6, "M"), (1e3, "K")):
        if n >= scale:
            value = n / scale
            return (f"{value:.1f}" if value < 100 else f"{value:.0f}").removesuffix(".0") + unit
    return f"{n:,}"


def duration(seconds):
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.3g} h"
    if seconds >= 60:
        return f"{seconds / 60:.3g} min"
    return f"{seconds:.3g} s"


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


def axis_style(ax, theme, title, subtitle, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=theme["text"], loc="left", pad=24)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5, color=theme["muted"],
            ha="left", va="bottom")
    ax.set_xlabel(xlabel, fontsize=9, color=theme["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=theme["muted"])
    ax.set_facecolor(theme["surface"])
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(theme["grid"])


def build_b1(rows, theme_name, out):
    """Largest sample the quotient finishes per homology dimension."""
    theme = THEMES[theme_name]
    color = theme["series"][QUOTIENT]
    found = reach(rows)
    dims = sorted(found)
    xs = list(range(len(dims)))
    ns = [found[d][0] for d in dims]
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    fig.patch.set_facecolor(theme["surface"])
    floor, top = min(ns) / 3, max(ns) * 3.5

    ax.fill_between(xs, floor, ns, color=color, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(xs, ns, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)
    for xi, dim in zip(xs, dims):
        n, stop = found[dim]
        if stop == "error":
            ax.plot([xi], [n], marker="X", markersize=8.5, markerfacecolor=color,
                    markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=4)
        else:
            filled = stop != "oom"
            ax.plot([xi], [n], marker="o", markersize=8.5,
                    markerfacecolor=color if filled else theme["surface"],
                    markeredgecolor=theme["surface"] if filled else color,
                    markeredgewidth=2, zorder=4)
        label = ("≥ " if stop == "cap" else "") + compact(n)
        ax.annotate(label, (xi, n), textcoords="offset points", xytext=(0, 9),
                    ha="center", va="bottom", fontsize=8, color=theme["text"], zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(floor, top)
    decades = [10.0 ** k for k in range(1, 8)]
    ticks = [t for t in decades if floor <= t <= top]
    if len(ticks) < 3:  # a narrow range: add the 2x and 5x steps
        ticks = [t * m for t in decades for m in (1, 2, 5) if floor <= t * m <= top]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([compact(int(t)) for t in ticks])
    ax.set_xticks(xs, [f"H{d}" for d in dims])
    ax.set_xlim(-0.35, len(dims) - 0.65)
    ax.grid(True, axis="y", which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0, pad=6)
    budget = f"{duration(rows[0]['timeout_s'])} and {float(rows[0]['mem_cap_gb']):g} GB"
    axis_style(ax, theme, "b1 · how far the quotient gets",
               rf"largest $S^d$ sample TDA's quotient finishes within {budget}, computed to"
               rf" $H_d$, all cores",
               "homology dimension", "points  (log scale)")

    # One series, so no legend box: the title names it. The key explains markers.
    stops = {stop for _, stop in found.values()}
    key = [Line2D([], [], linestyle="none", marker="o", markersize=7.5,
                  markerfacecolor=theme["muted"], markeredgecolor=theme["surface"],
                  markeredgewidth=1.5, label="stopped by time")]
    if "oom" in stops:
        key.append(Line2D([], [], linestyle="none", marker="o", markersize=7.5,
                          markerfacecolor=theme["surface"], markeredgecolor=theme["muted"],
                          markeredgewidth=1.8, label="stopped by memory"))
    if "error" in stops:
        key.append(Line2D([], [], linestyle="none", marker="X", markersize=7.5,
                          markerfacecolor=theme["muted"], markeredgecolor=theme["surface"],
                          label="stopped by an error"))
    ax.legend(handles=key, frameon=False, fontsize=8, labelcolor=theme["muted"],
              loc="upper right", bbox_to_anchor=(1.0, 0.97), handletextpad=0.4)
    fig.tight_layout()
    save(fig, theme, out)


def build_b2(rows, theme_name, out):
    """Filtration size without and with the quotient, one row per dataset."""
    theme = THEMES[theme_name]
    plain, color = theme["plain"], theme["series"][QUOTIENT]
    sizes, points = defaultdict(dict), {}
    for row in rows:
        if row["status"] == "ok":
            sizes[row["dataset"]][row["pipeline"]] = int(row["simplices"])
            points[row["dataset"]] = int(row["n"])
    names = sorted(sizes, key=lambda name: max(sizes[name].values()))  # largest on top
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(theme["surface"])
    cell = ax.get_yaxis_transform()

    for y, name in enumerate(names):
        s = sizes[name]
        if "vanilla" in s and "quotient" in s:
            ax.plot([s["quotient"], s["vanilla"]], [y, y], color=theme["grid"], linewidth=2.5,
                    solid_capstyle="round", zorder=1)
            ax.text(1.1, y, f"{compact(round(s['vanilla'] / s['quotient']))}×", transform=cell,
                    ha="right", va="center", fontsize=8.5, color=theme["text"])
        # Each count sits just outside its end, in muted ink: the factor is the headline.
        # The surface behind it hides the gridlines it crosses.
        tag = dict(textcoords="offset points", va="center", fontsize=7.5, color=theme["muted"],
                   bbox=dict(boxstyle="square,pad=0.15", facecolor=theme["surface"],
                             edgecolor="none"))
        if "vanilla" in s:
            ax.plot([s["vanilla"]], [y], linestyle="none", marker="|", markersize=15,
                    markeredgecolor=plain, markeredgewidth=2.5, zorder=3)
            ax.annotate(compact(s["vanilla"]), (s["vanilla"], y), xytext=(7, 0), ha="left",
                        **tag)
        if "quotient" in s:
            ax.plot([s["quotient"]], [y], linestyle="none", marker="o", markersize=9,
                    markerfacecolor=color, markeredgecolor=theme["surface"],
                    markeredgewidth=2, zorder=3)
            ax.annotate(compact(s["quotient"]), (s["quotient"], y), xytext=(-8, 0), ha="right",
                        **tag)
    ax.text(1.1, len(names) - 0.45, "× smaller", transform=cell, ha="right", va="bottom",
            fontsize=8, color=theme["muted"])

    every = [v for s in sizes.values() for v in s.values()]
    low, high = min(every) / 6, max(every) * 6  # room for the end labels
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ticks = [10 ** k for k in range(1, 13) if low <= 10 ** k <= high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([compact(t).replace("000M", "B") for t in ticks])
    ax.set_yticks(range(len(names)), [f"{label(name)}  ·  {points[name]:,}" for name in names])
    ax.set_ylim(-0.6, len(names) + 0.45)
    ax.grid(True, axis="x", which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=8, length=0, pad=6)
    ax.tick_params(axis="y", colors=theme["text"], labelsize=8.5, length=0, pad=8)
    axis_style(ax, theme, "b2 · how much the quotient shrinks the filtration",
               f"simplices up to dimension {rows[0]['max_dim']} per dataset (points);"
               " neither side peels or collapses",
               "simplices in the filtration  (log scale)", None)

    handles = [
        Line2D([], [], linestyle="none", marker="|", markersize=11, markeredgecolor=plain,
               markeredgewidth=2.2, label="without the quotient"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor=color,
               markeredgecolor=theme["surface"], markeredgewidth=1.8,
               label="with the quotient"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=theme["muted"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.06, 0.9, 1))
    save(fig, theme, out)


def label(dataset):
    """Display name: `klein_400` reads as Klein bottle; the size is shown apart."""
    return DATASETS.get(dataset.split("_")[0], dataset)


def save(fig, theme, out):
    for suffix in ("svg", "png"):
        fig.savefig(f"{out}.{suffix}", dpi=200, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out}.svg and {out}.png")


def build(a_rows, b_rows, c_rows, theme_name, out):
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.patch.set_facecolor(theme["surface"])

    a_title, b_title, c_title = titles(a_rows, b_rows, c_rows)
    timing_panel(axes[0], a_rows, "a", "n", theme, a_title, "n  (points)", logx=True)
    timing_panel(axes[1], b_rows, "b", "max_dim", theme, b_title, "max_dim", logx=False)
    error_panel(axes[2], c_rows, theme, c_title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=theme["muted"],
               loc="upper left", ncol=len(labels), bbox_to_anchor=(0.006, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("svg", "png"):
        fig.savefig(f"{out}.{suffix}", dpi=200, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out}.svg and {out}.png")


def main():
    panels = {p: read(RESULTS / f"{p}.csv") for p in ("a", "b", "c", "b1", "b2")}
    for panel, rows in panels.items():
        print(f"panel {panel}: {len(rows)} rows" + ("" if rows else "  (not run yet)"))
    if not any(panels.values()):
        raise SystemExit(f"no results in {RESULTS}; run figure2.py first")

    for theme, suffix in (("light", ""), ("dark", "-dark")):
        if panels["a"] or panels["b"] or panels["c"]:
            build(panels["a"], panels["b"], panels["c"], theme,
                  RESULTS / f"figure2{suffix}")
        if reach(panels["b1"]):
            build_b1(panels["b1"], theme, RESULTS / f"figure2-b1{suffix}")
        if any(row["status"] == "ok" for row in panels["b2"]):
            build_b2(panels["b2"], theme, RESULTS / f"figure2-b2{suffix}")


if __name__ == "__main__":
    main()
