"""Aggregate-only figures for docs/images (no store-level rows are read: inputs are evals/results/*.json)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "evals" / "results", ROOT / "docs" / "images"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"model": "#2a78d6", "baseline": "#eb6834", "snaive": "#1baf7a"}  # validated categorical slots 1-3
NAMES = {"model": "Global LightGBM", "baseline": "Promo-aware weekday mean (K=12)", "snaive": "Seasonal naive 14"}
HB = ["1-7", "8-14", "15-21", "22-28", "29-35", "36-42"]


def j(name):
    return json.loads((RES / name).read_text())


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def fig(w=8, h=4.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    f.patch.set_facecolor(SURFACE)
    return f, ax


def save(f, name):
    f.tight_layout()
    f.savefig(OUT / name, facecolor=SURFACE)
    plt.close(f)


def test_metric(t, key, m="wape"):
    return t["variants"][key]["overall"]["views"]["test"][m]


def main():
    t = j("final_test.json")
    # 1. TEST comparison (WAPE with 95% CI)
    f, ax = fig(8, 3.4)
    keys = [("snaive", "snaive14"), ("baseline", "baseline"), ("model", "full")]
    for i, (k, v) in enumerate(keys):
        c = test_metric(t, v)
        ax.barh(i, c["value"], height=0.5, color=C[k])
        ax.errorbar(c["value"], i, xerr=[[c["value"] - c["lo"]], [c["hi"] - c["value"]]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text(c["hi"] + 0.002, i, f"{c['value']:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(3), [NAMES[k] for k, _ in keys])
    style(ax, "TEST window: WAPE with 95% store-bootstrap CI (lower is better)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    save(f, "test_comparison.png")
    # 2. Error by horizon on TEST
    f, ax = fig()
    for k, v in (("model", "full"), ("baseline", "baseline"), ("snaive", "snaive14")):
        ys = [t["variants"][v]["horizon"]["views"]["test"][b]["wape"]["value"] for b in HB]
        ax.plot(range(6), ys, color=C[k], lw=2, marker="o", ms=5, label=NAMES[k])
        ax.text(5.1, ys[-1], f"{ys[-1]:.3f}", color=INK2, fontsize=8, va="center")
    ax.set_xticks(range(6), HB)
    ax.set_xlabel("days after cutoff", color=INK2, fontsize=9)
    style(ax, "TEST: WAPE by forecast horizon", "WAPE")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(f, "test_error_by_horizon.png")
    # 3. Interval coverage, uncalibrated vs calibrated
    ic = j("intervals_calibrated.json")
    sets = [("Backtest", "backtest_uncalibrated", "backtest_calibrated_in_sample"), ("VALIDATION", "validation_uncalibrated", "validation_calibrated"),
            ("TEST", "test_uncalibrated", "test_calibrated")]
    f, ax = fig(8, 4)
    w = 0.36
    for i, (lbl, u, c) in enumerate(sets):
        for off, key, col, name in ((-w / 2, u, "#8fb6ea", "uncalibrated"), (w / 2, c, C["model"], "calibrated (backtest factors)")):
            v = ic[key]["overall"]["coverage"]["value"]
            ax.bar(i + off, v, width=w - 0.03, color=col, label=name if i == 0 else None)
            ax.text(i + off, v + 0.005, f"{v:.3f}", ha="center", color=INK, fontsize=8)
    ax.axhline(0.8, color=INK2, lw=1, ls="--")
    ax.text(2.55, 0.802, "nominal 0.80", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(3), [s[0] for s in sets])
    ax.set_ylim(0.6, 0.9)
    style(ax, "80% interval coverage (backtest calibrated values are in-sample)", "share of actuals inside the interval")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=2)
    save(f, "interval_coverage.png")
    # 4. Backtest vs VALIDATION vs TEST, model vs baseline
    mb = j("model_backtest.json")["variants"]
    mv = j("model_validation.json")["variants"]
    pts = {"model": [mb["full"]["overall"]["views"]["all_folds"]["wape"]["value"], mv["full"]["overall"]["views"]["validation"]["wape"]["value"],
                     test_metric(t, "full")["value"]],
           "baseline": [mb["baseline"]["overall"]["views"]["all_folds"]["wape"]["value"], mv["baseline"]["overall"]["views"]["validation"]["wape"]["value"],
                        test_metric(t, "baseline")["value"]]}
    f, ax = fig(8, 4)
    for off, k in ((-w / 2, "model"), (w / 2, "baseline")):
        for i, v in enumerate(pts[k]):
            ax.bar(i + off, v, width=w - 0.03, color=C[k], label=NAMES[k] if i == 0 else None)
            ax.text(i + off, v + 0.002, f"{v:.3f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(3), ["Backtest (mean of 8 folds)", "VALIDATION", "TEST"])
    style(ax, "WAPE across evaluation windows (lower is better)", "WAPE")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(f, "validation_vs_test.png")
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    sys.exit(main())
