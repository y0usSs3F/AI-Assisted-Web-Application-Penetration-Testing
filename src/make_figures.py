"""
make_figures.py
Render every figure in the paper from the saved result files (no retraining).

Style rules (applied throughout): detectors keep a fixed colour from a colour-blind-validated
categorical order, with the conventional baseline in neutral grey; magnitudes use a single-hue
blue ramp; ordered settings use an ordinal blue ramp; hairline solid grids; line charts carry
marker shapes as a second cue so the figures survive black-and-white printing.

Run after all experiments:   python make_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve

import config
import data as D

# ------------------------------------------------------------------ style
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # validated categorical order
COLOR = dict(zip(config.ML_MODELS, SLOTS)) | {config.BASELINE: MUTED}
MARKER = {"logreg": "o", "linsvm": "s", "rf": "^", "hgb": "D", "mlp": "v", config.BASELINE: "X"}
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]                     # validated ordinal ramp
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                    "#1c5cab", "#0d366b"])
ALL = config.ML_MODELS + [config.BASELINE]
S = config.SHORT

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "figure.facecolor": "white", "axes.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
})


def _load(name):
    return json.loads((config.RESULTS_DIR / name).read_text())


def _save(fig, name):
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ figures
def fig01_data():
    p = _load("data_profile.json")["thursday"]
    stages = ["Raw flows", "Finite values", "Unique flows", "HTTP flows (port 80)"]
    counts = [p["raw_rows"], sum(p["after_nonfinite"].values()), p["after_dedup_rows"], p["http_rows"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(stages))[::-1]
    ax[0].barh(y, counts, color=SLOTS[0], height=0.55)
    ax[0].set_yticks(y, stages); ax[0].set_xlabel("flows"); ax[0].grid(axis="y", visible=False)
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax[0].set_title("Thursday-morning traffic: cleaning and scope")
    ax[0].set_xlim(0, max(counts) * 1.25)
    for yi, c in zip(y, counts):
        ax[0].text(c + max(counts) * 0.015, yi, f"{c:,}", va="center", fontsize=9, color=INK2)
    lab = p["http_labels"]
    cls = config.CLASSES
    vals = [lab.get(c, 0) for c in cls]
    y2 = np.arange(len(cls))[::-1]
    ax[1].barh(y2, vals, color=SLOTS[0], height=0.55)
    ax[1].set_xscale("log"); ax[1].set_xlim(10, 1e5)
    ax[1].set_yticks(y2, cls); ax[1].set_xlabel("flows (log scale)"); ax[1].grid(axis="y", visible=False)
    ax[1].set_title("Classes in the HTTP-flow dataset")
    for yi, v in zip(y2, vals):
        ax[1].text(v * 1.12, yi, f"{v:,}", va="center", fontsize=9, color=INK2)
    _save(fig, "fig01_data.png")


def fig02_detect_vs_classify():
    cv = _load("metrics.json")["cv"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(ALL))[::-1]
    for k, (task, key, title) in enumerate([("binary", "f1", "Detection: attack vs. benign (F1)"),
                                             ("multiclass", "macro_f1", "Classification: which attack (macro-F1)")]):
        for yi, n in zip(y, ALL):
            if n not in cv[task]:
                ax[k].text(0.02, yi, "n/a: rules cannot name the attack type", va="center", fontsize=8.5, color=MUTED)
                continue
            m, s = cv[task][n][f"{key}_mean"], cv[task][n][f"{key}_std"]
            ax[k].barh(yi, m, xerr=s, color=COLOR[n] if n == config.BASELINE else SLOTS[0], height=0.55,
                       error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
            ax[k].text(max(m + s, 0) + 0.015, yi, f"{m:.3f}", va="center", fontsize=9, color=INK2)
        ax[k].set_xlim(0, 1.12); ax[k].set_title(title); ax[k].grid(axis="y", visible=False)
        ax[k].set_xlabel("score (5-fold CV mean ± std)")
    ax[0].set_yticks(y, [config.DISPLAY[n] for n in ALL])
    _save(fig, "fig02_detect_vs_classify.png")


def fig03_roc():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for n in ALL:
        d = np.load(config.PRED_DIR / f"binary_{n}.npz", allow_pickle=True)
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        auc = _load("metrics.json")["binary"][n]["roc_auc"]
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=COLOR[n], lw=2, marker=MARKER[n], markevery=0.12,
                ms=5, label=f"{S[n]} (AUC {auc:.3f})")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(-0.02, 1.02)
    ax.axvline(0.01, color=AXIS, lw=1)
    ax.text(0.0112, 0.62, "1% false-alarm budget", fontsize=8.5, color=MUTED, rotation=90, va="center")
    ax.set_xlabel("false-positive rate (log scale)"); ax.set_ylabel("detection rate (TPR)")
    ax.set_title("ROC curves, held-out test (HTTP flows)")
    ax.legend(loc="center right")
    _save(fig, "fig03_roc.png")


def fig04_threshold_ids():
    b = _load("metrics.json")["binary"][config.BASELINE]
    d = np.load(config.PRED_DIR / f"binary_{config.BASELINE}.npz", allow_pickle=True)
    s, y = d["y_score"].astype(int), d["y_true"]
    xs = np.arange(0, max(int(s.max()), b["alarm_threshold_m"]) + 2)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    for off, cls, col, lab in [(-w / 2, 0, SLOTS[0], "benign flows"), (w / 2, 1, SLOTS[1], "attack flows")]:
        share = np.array([(s[y == cls] == v).mean() for v in xs])
        ax.bar(xs + off, np.where(share > 0, share, np.nan), width=w, color=col, label=lab,
               edgecolor=SURFACE, linewidth=1)
    m = b["alarm_threshold_m"]
    ax.axvline(m - 0.5, color=INK2, lw=1.2)
    ax.text(m - 0.35, 0.3, f"alarm threshold: ≥{m} metrics\n(keeps false alarms ≤ 1%)",
            fontsize=8.5, color=INK2, va="top")
    ax.set_yscale("log"); ax.set_ylim(2e-4, 1.5)
    ax.set_xticks(xs); ax.set_xlabel("anomaly score = number of flow metrics outside the benign range")
    ax.set_ylabel("share of flows (log)"); ax.grid(axis="x", visible=False)
    ax.set_title("Why the conventional threshold IDS misses every attack")
    ax.legend(loc="upper right")
    _save(fig, "fig04_threshold_ids_scores.png")


def fig05_validity():
    va = _load("validity_audit.json")["settings"]
    keys = ["S0", "S1", "S2", "S3"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"wspace": 0.18})
    x = np.arange(len(config.ML_MODELS)); w = 0.2
    for k, (part, metric, title) in enumerate([("binary", "f1", "Detection F1"),
                                               ("multiclass", "macro_f1", "Attack-type macro-F1")]):
        for i, s in enumerate(keys):
            vals = [va[s][part][n][metric] for n in config.ML_MODELS]
            lab = f"{s}: {va[s]['description']} ({int(sum(v for c, v in va[s]['counts'].items() if c != config.BENIGN))} attacks)"
            ax[k].bar(x + (i - 1.5) * w, vals, width=w, color=ORDINAL[i], edgecolor=SURFACE, linewidth=1,
                      label=lab)
        ax[k].set_xticks(x, [S[n] for n in config.ML_MODELS]); ax[k].set_ylim(0, 1.05)
        ax[k].set_title(title); ax[k].grid(axis="x", visible=False)
    ax[0].set_ylabel("5-fold CV (pooled out-of-fold)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    _save(fig, "fig05_validity_audit.png")


def fig06_confusion():
    cv = _load("metrics.json")["cv"]["multiclass"]
    va = _load("validity_audit.json")["settings"]
    cls = config.CLASSES
    panels = [("All HTTP flows (S0)", np.array(cv["hgb"]["oof_confusion"])),
              ("Payload-bearing flows only (S2)", np.array(va["S2"]["multiclass"]["hgb"]["confusion"]))]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.45})
    for a, (title, cm) in zip(ax, panels):
        rn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = a.imshow(rn, cmap=BLUES, vmin=0, vmax=1)
        a.set_xticks(range(len(cls)), cls, rotation=25, ha="right"); a.set_yticks(range(len(cls)), cls)
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.grid(False)
        a.set_title(f"Gradient-Boosted Trees: {title}")
        for i in range(len(cls)):
            for j in range(len(cls)):
                a.text(j, i, f"{rn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                       color="white" if rn[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, label="share of true class (recall)")
    _save(fig, "fig06_confusion.png")


def fig07_unseen():
    ua = _load("unseen_attacks.json")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(ALL))[::-1]
    for a, att in zip(ax, config.ATTACKS):
        for yi, n in zip(y, ALL):
            seen, unseen = ua[att][n]["seen_recall_oof"], ua[att][n]["unseen_recall"]
            a.plot([seen, unseen], [yi, yi], color=AXIS, lw=2, zorder=1)
            a.scatter(seen, yi, s=55, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=2,
                      label="attack type seen in training" if yi == y[0] else None)
            a.scatter(unseen, yi, s=55, color=SLOTS[0], zorder=3,
                      label="attack type never seen" if yi == y[0] else None)
        a.set_xlim(-0.05, 1.05); a.grid(axis="y", visible=False)
        a.set_title(f"{att} ({ua[att]['n_flows']} flows)"); a.set_xlabel("detection rate")
    ax[0].set_yticks(y, [config.DISPLAY[n] for n in ALL])
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    _save(fig, "fig07_unseen_attacks.png")


def fig08_timing():
    ev = _load("evasion_timing.json")
    ks = ev["factors"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for n in ALL:
        r = [ev["recall"][n]["overall"][str(k)] for k in ks]
        ax.plot(ks, r, color=COLOR[n], lw=2, marker=MARKER[n], ms=6, label=config.DISPLAY[n])
    ax.set_xscale("log"); ax.set_xticks(ks, [f"{k}×" for k in ks]); ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("attacker slow-down factor k (time features ×k, rates ÷k)")
    ax.set_ylabel("detection rate on attack flows")
    ax.set_title(f"Low-and-slow timing evasion ({ev['n_attacks']} held-out attack flows)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, "fig08_timing_evasion.png")


def fig09_cross_day():
    cd = _load("cross_day.json")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(ALL))[::-1]
    for yi, n in zip(y, ALL):
        t, m = 100 * cd[n]["thursday_fpr"], 100 * cd[n]["monday_fpr"]
        ax.plot([t, m], [yi, yi], color=AXIS, lw=2, zorder=1)
        ax.scatter(t, yi, s=55, color=SLOTS[0], zorder=2, label="Thursday (test split)" if yi == y[0] else None)
        ax.scatter(m, yi, s=55, color=SLOTS[1], marker="s", zorder=3,
                   label="Monday (unseen benign day)" if yi == y[0] else None)
        ax.text(max(t, m) + 0.07, yi, f"{cd[n]['monday_false_alerts']:,} false alerts on Monday",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [config.DISPLAY[n] for n in ALL]); ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.1, 4.2); ax.set_xlabel("false-positive rate (%)")
    ax.set_title(f"False alarms on a new day ({cd['rf']['monday_flows']:,} benign Monday HTTP flows)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, "fig09_cross_day.png")


def fig10_importance():
    fi = _load("metrics.json")["feature_importance"]
    fp = {"Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward"}
    imp = fi["impurity"][:12][::-1]
    per = fi["permutation"][:8][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.75})
    ax[0].barh([n for n, _ in imp], [v for _, v in imp], height=0.6,
               color=[SLOTS[1] if n in fp else SLOTS[0] for n, _ in imp])
    ax[0].set_title("Impurity importance (top 12)"); ax[0].set_xlabel("mean decrease in impurity")
    ax[0].grid(axis="y", visible=False)
    ax[0].legend(handles=[Patch(color=SLOTS[0], label="traffic behaviour feature"),
                          Patch(color=SLOTS[1], label="TCP-stack fingerprint feature")],
                 loc="upper center", bbox_to_anchor=(0.3, -0.16), ncol=2)
    ax[1].barh([n for n, _, _ in per], [v for _, v, _ in per], xerr=[s for _, _, s in per], height=0.6,
               color=SLOTS[0], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax[1].set_title("Permutation importance (top 8)"); ax[1].set_xlabel("drop in average precision")
    ax[1].grid(axis="y", visible=False)
    _save(fig, "fig10_feature_importance.png")


def main():
    for fn in (fig01_data, fig02_detect_vs_classify, fig03_roc, fig04_threshold_ids, fig05_validity,
               fig06_confusion, fig07_unseen, fig08_timing, fig09_cross_day, fig10_importance):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skipped {fn.__name__}: missing {e.filename}")


if __name__ == "__main__":
    main()
