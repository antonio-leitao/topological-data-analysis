"""Draw Figure 2 from benchmark CSVs."""

import csv
from collections import defaultdict
from math import log
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
ENGINES = ("giotto", "gudhi", "tda", "tda_quotient")
LABELS = {"giotto": "giotto-ph", "gudhi": "GUDHI",
          "tda": "TDA (exact)", "tda_quotient": "TDA (quotient)"}

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#898781", grid="#e1e0d9",
                  series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100")),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#898781", grid="#2c2c2a",
                 series=("#3987e5", "#d95926", "#199e70", "#c98500")),
}


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
        a += f"   ({symbol(a_rows[0])}{superscript(a_rows[0]['dim'])}"
        a += f", max_dim {a_rows[0]['max_dim']})"
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
    color = theme["series"][3]

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
    panels = {p: read(RESULTS / f"{p}.csv") for p in "abc"}
    for panel, rows in panels.items():
        print(f"panel {panel}: {len(rows)} rows" + ("" if rows else "  (not run yet)"))
    if not any(panels.values()):
        raise SystemExit(f"no results in {RESULTS}; run figure2.py first")

    for theme, suffix in (("light", ""), ("dark", "-dark")):
        build(panels["a"], panels["b"], panels["c"], theme,
              RESULTS / f"figure2{suffix}")


if __name__ == "__main__":
    main()
