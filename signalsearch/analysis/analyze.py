"""Batch analysis: reads every seed's eval.json / report.json / trace.jsonl and writes charts + summary.md.

Optional `compare` batches (e.g. the same seeds with a different k) produce a sensitivity table.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# reference palette (light mode): categorical slots in fixed order, status colours reserved for outcomes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"rescued": "#0ca30c", "step_cap": "#fab219", "battery_depleted": "#ec835a", "target_died": "#d03b3b"}
MARKER = {"rescued": "o", "step_cap": "s", "battery_depleted": "D", "target_died": "X"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPEND_ORDER = ["move", "turn", "lidar", "radar", "thermal", "cv"]
KINDS = ["target", "dog", "torch", "hearth", "armour", "none"]


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_batch(batch_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(Path(batch_dir).glob("seed*")):
        if not (d / "eval.json").exists():
            continue
        ev = json.loads((d / "eval.json").read_text())
        rep = json.loads((d / "report.json").read_text())
        cfg = yaml.safe_load((d / "config.resolved.yaml").read_text())
        ev.update(seed=cfg["seed"], dir=d.name, k=cfg["world"]["target"]["k"], report=rep)
        runs.append(ev)
    return runs


def _batch_config(batch_dir) -> str:
    p = Path(batch_dir) / "batch.json"
    if not p.exists():
        return "?"
    b = json.loads(p.read_text())
    w = b.get("weather", {})
    sets = " ".join(f"--set {x}" for x in b.get("sets", []))
    return (f"`{b['config']}{' ' + sets if sets else ''}`, weather {w.get('temperature')} C / {w.get('humidity')}% RH / "
            f"rain {w.get('rain')} mm ({w.get('origin', '')}),")


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return name


def chart_outcomes(runs, out):
    c = Counter(r["outcome"] for r in runs)
    order = [k for k in ("rescued", "target_died", "battery_depleted", "step_cap") if c.get(k)]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    y = np.arange(len(order))
    ax.barh(y, [c[k] for k in order], color=[STATUS[k] for k in order], height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [k.replace("_", " ") for k in order])
    ax.invert_yaxis()
    for i, k in enumerate(order):
        ax.text(c[k] + 0.2, i, f"{c[k]}  ({100 * c[k] / len(runs):.0f}%)", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(c.values()) * 1.35)
    _style(ax, f"Mission outcomes ({len(runs)} missions)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "outcomes.png")


def chart_turns_vs_life(runs, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lim = max(max(r["life_turns"] for r in runs), max(r["turns"] for r in runs)) * 1.08
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--")
    ax.text(lim * 0.97, lim * 0.93, "turns = life", ha="right", color=INK2, fontsize=8)
    for oc in STATUS:
        rs = [r for r in runs if r["outcome"] == oc]
        if rs:
            ax.scatter([r["life_turns"] for r in rs], [r["turns"] for r in rs], s=46, marker=MARKER[oc],
                       color=STATUS[oc], edgecolor=SURFACE, linewidth=1.5, label=oc.replace("_", " "), zorder=3)
    for r in runs:
        ax.annotate(str(r["seed"]), (r["life_turns"], r["turns"]), xytext=(4, -9), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mission life budget (turns)", color=INK2)
    ax.set_ylabel("turns used", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Turns used vs. the target's life budget")
    return _save(fig, out, "turns_vs_life.png")


def chart_spend(runs, out):
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(runs) + 1.4))
    y = np.arange(len(runs))
    left = np.zeros(len(runs))
    for i, cat in enumerate(SPEND_ORDER):
        v = np.array([r["spend"].get(cat, 0.0) for r in runs])
        if v.sum() == 0:
            continue
        ax.barh(y, v, left=left, color=SERIES[i], height=0.7, edgecolor=SURFACE, linewidth=1.5, label=cat)
        left += v
    ax.set_yticks(y, [f"seed {r['seed']}" + (f"  +{r['recharges']} chg" if r["recharges"] else "") for r in runs],
                  fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("battery spent (%)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07 * 20 / max(len(runs), 1)))
    _style(ax, "Battery spend by category")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "battery_spend.png")


def chart_mass(runs, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    grid = np.linspace(0, 1, 101)
    groups = {"rescued": [], "failed": []}
    for r in runs:
        if not r["mass_at_truth"]:
            continue
        t = np.array([m[0] for m in r["mass_at_truth"]], dtype=float) / r["life_turns"]
        m = np.array([m[1] for m in r["mass_at_truth"]])
        g = "rescued" if r["outcome"] == "rescued" else "failed"
        col = SERIES[0] if g == "rescued" else SERIES[1]
        ax.plot(t, m, color=col, alpha=0.25, linewidth=1)
        groups[g].append(np.interp(grid, t, m, right=np.nan))
    for g, col in (("rescued", SERIES[0]), ("failed", SERIES[1])):
        if groups[g]:
            arr = np.array(groups[g])
            ok = (~np.isnan(arr)).sum(axis=0) >= max(2, len(arr) / 2)  # only while most missions still run
            med = np.full(len(grid), np.nan)
            med[ok] = np.nanmedian(arr[:, ok], axis=0)
            ax.plot(grid, med, color=col, linewidth=2.2, label=f"{g} (median of {len(groups[g])})")
    ax.set_xlabel("time as a fraction of the target's life", color=INK2)
    ax.set_ylabel("belief mass within 2 cells of the truth", color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "How fast the belief found the target")
    return _save(fig, out, "belief_mass_at_truth.png")


def _first_time(r, thr):
    for t, m in r["mass_at_truth"]:
        if m >= thr:
            return t
    return None


def _suggestions(runs, rate, life_frac_rescued, causes, spend_mean, decoy_fooled, fooled, sweep) -> list[str]:
    s = []
    late = causes.get("localized but never CV-confirmed", 0) + causes.get("confirmed but could not reach the target in time", 0)
    never = causes.get("never localized the target", 0)
    failures = len(runs) - int(round(rate * len(runs)))
    if sweep and len(sweep) > 1:
        lo, hi = sweep[0], sweep[-1]
        if hi[1] - lo[1] <= 0.1 * len(runs):
            s.append(f"**Life budget (k) is not the bottleneck.** Raising k from {lo[0]} to {hi[0]} moved rescues only "
                     f"from {lo[1]} to {hi[1]} of {len(runs)}: the {late} \"localized but never confirmed\" failures "
                     "are the agent stalling near the target (belief cluster offset from the true cell, approach "
                     "routes through unmapped walls), not a short clock. Fix the confirmation step before touching k.")
        else:
            best = min((k for k, n in sweep if n == hi[1]), default=hi[0])
            plateau = f" and plateau from k = {best}" if best < hi[0] else ""
            s.append(f"**Life budget (k).** Rescues rise from {lo[1]} to {hi[1]} of {len(runs)} as k goes {lo[0]} -> "
                     f"{hi[0]}{plateau}; rescued missions used a median {np.median(life_frac_rescued) * 100:.0f}% of "
                     f"the budget. k = {lo[0]} is tight for an agent that must map the castle as it goes; the "
                     f"{len(runs) - hi[1]} missions still lost with extra time are stalls, not a short clock.")
    elif failures and late:
        s.append(f"**Life budget (k).** {late} of {failures} failures had already found the target; rescued missions "
                 f"used a median {np.median(life_frac_rescued) * 100:.0f}% of the budget. Run a k sweep "
                 "(`batch --set world.target.k=...` then `analyze --compare`) to see whether time or stalling is to blame.")
    if never:
        s.append(f"**Early search.** {never} missions never put 10% of the belief near the target. The distress-call "
                 "cone (+-25 deg) is wide at long range; a wider first thermal sweep or a second sweep from the cone's far "
                 "end would help.")
    if spend_mean.get("move", 0) > 0 and np.mean([r["recharges"] for r in runs]) < 0.2:
        s.append("**Battery is rarely binding.** Almost no mission needed a recharge; the clock, not the battery, "
                 "decides most missions. A smaller battery or pricier sensors would make the charging policy matter.")
    worst = max(((k, v) for k, v in decoy_fooled.items()), key=lambda kv: kv[1], default=(None, 0))
    if worst[1]:
        by = max(fooled, key=lambda sn: fooled[sn].get(worst[0], 0))
        why = {"thermal": "a warm source behind unknown walls is hard to tell from a person by strength alone",
               "radar": "a heartbeat is a heartbeat: only LIDAR shape or CV separates it from the target",
               "lidar_humanoid": "LIDAR shape confusion turns it into a humanoid"}[by]
        s.append(f"**Decoys.** The {worst[0]} fooled the belief most often ({worst[1]} times, mostly via {by}: {why}). "
                 f"Its assumed signature count (agent_assumed.signature_counts) is the first knob to tune.")
    return s


def analyze(batch_dir, out_dir, compare: list | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_batch(batch_dir)
    if not runs:
        raise SystemExit(f"no finished runs in {batch_dir}")
    n = len(runs)
    oc = Counter(r["outcome"] for r in runs)
    rate = oc["rescued"] / n
    charts = [chart_outcomes(runs, out), chart_turns_vs_life(runs, out), chart_spend(runs, out), chart_mass(runs, out)]
    rescued = [r for r in runs if r["outcome"] == "rescued"]
    life_frac = [r["turns"] / r["life_turns"] for r in rescued]
    causes = Counter(r["failure_cause"] for r in runs if r["failure_cause"])
    spend_mean = {k: float(np.mean([r["spend"].get(k, 0) for r in runs])) for k in SPEND_ORDER}
    sensors = ("thermal", "radar", "lidar_humanoid", "cv_target")
    readings = {s: {k: sum(r["readings_by_source"][s][k] for r in runs) for k in KINDS} for s in sensors}
    fooled = {s: {k: sum(r["fooled"].get(s, {}).get(k, 0) for r in runs) for k in KINDS[:-1]} for s in ("thermal", "radar", "lidar_humanoid")}
    decoy_fooled = Counter()
    for s in fooled:
        for k, v in fooled[s].items():
            if k != "target":
                decoy_fooled[k] += v
    cv = {k: Counter() for k in KINDS[:-1]}
    for r in runs:
        for k, labs in r["cv_labels_by_true_kind"].items():
            cv[k].update(labs)
    t10 = [_first_time(r, 0.1) for r in runs]
    t50 = [_first_time(r, 0.5) for r in runs]
    L = []
    L += [f"# 20-game analysis: `{Path(batch_dir).name}`", "",
          f"Config: {_batch_config(batch_dir)} seeds {runs[0]['seed']}-{runs[-1]['seed']}. "
          f"Generated by `python -m signalsearch analyze`.", "",
          "## Success rate", "",
          f"**{oc['rescued']}/{n} rescued ({100 * rate:.0f}%).** " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in oc.items()), "",
          f"![outcomes]({charts[0]})", "",
          "## Turns used vs. each mission's life (is k = 1.3 right?)", "",
          f"![turns vs life]({charts[1]})", ""]
    if rescued:
        L.append(f"Rescued missions used a median **{np.median(life_frac) * 100:.0f}%** of the life budget "
                 f"(range {min(life_frac) * 100:.0f}-{max(life_frac) * 100:.0f}%). "
                 f"Median turns for the belief to put 10% / 50% of its mass within 2 cells of the target: "
                 f"{np.median([t for t in t10 if t is not None]) if any(t is not None for t in t10) else '-'} / "
                 f"{np.median([t for t in t50 if t is not None]) if any(t is not None for t in t50) else '-'}.")
    L += ["", "| seed | outcome | turns | life | used | walking dist | first 10% mass | first 50% mass | failure cause |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r, a, b in zip(runs, t10, t50):
        L.append(f"| {r['seed']} | {r['outcome']} | {r['turns']} | {r['life_turns']} | {r['turns'] / r['life_turns']:.2f} | "
                 f"{r['d_path']} | {a if a is not None else '-'} | {b if b is not None else '-'} | {r['failure_cause'] or ''} |")
    sweep = []
    if compare:
        L += ["", "### k sensitivity (same seeds, only k changed)", "", "| k | rescued | median life used (rescued) |", "|---|---|---|"]
        for bd in [batch_dir] + list(compare):
            rs = load_batch(bd)
            ok = [r for r in rs if r["outcome"] == "rescued"]
            med = f"{np.median([r['turns'] / r['life_turns'] for r in ok]) * 100:.0f}%" if ok else "-"
            L.append(f"| {rs[0]['k']} | {len(ok)}/{len(rs)} | {med} |")
            sweep.append((rs[0]["k"], len(ok)))
        sweep.sort()
    L += ["", "## Battery spend by category", "", f"![spend]({charts[2]})", "",
          "Mean per mission: " + ", ".join(f"{k} {v:.1f}%" for k, v in spend_mean.items()) +
          f". Recharges per mission: mean {np.mean([r['recharges'] for r in runs]):.2f}, "
          f"max {max(r['recharges'] for r in runs)}.", "",
          "## Decoys: what each sensor actually saw", "",
          "Readings attributed by the evaluator to the nearest true entity at that turn (`none` = no entity within "
          "the sensor's error radius, i.e. noise).", "",
          "| sensor | " + " | ".join(KINDS) + " |", "|---" * (len(KINDS) + 1) + "|"]
    for s in sensors:
        L.append(f"| {s} | " + " | ".join(str(readings[s][k]) for k in KINDS) + " |")
    L += ["", "**Fooled** = within 5 turns of the reading the belief peak sat within 2 cells of that decoy:", "",
          "| sensor | " + " | ".join(KINDS[1:-1]) + " |", "|---" * (len(KINDS) - 1) + "|"]
    for s in fooled:
        L.append(f"| {s} | " + " | ".join(str(fooled[s][k]) for k in KINDS[1:-1]) + " |")
    L += ["", "**CV verdicts** (how often CV caught a decoy for what it is):", "",
          "| true entity | CV labels |", "|---|---|"]
    for k, c in cv.items():
        if c:
            L.append(f"| {k} | " + ", ".join(f"{lab}: {v}" for lab, v in c.most_common()) + " |")
    L += ["", "## Belief concentration on the true location", "", f"![mass]({charts[3]})", "",
          "## Endgame overrides", ""]
    ov_rows = [(r["seed"], o, r["outcome"]) for r in runs for o in r["overrides"]]
    if not ov_rows:
        L.append("None fired.")
    else:
        fired = Counter(seed for seed, _, _ in ov_rows)
        L.append(f"{len(ov_rows)} override evaluations in {len(fired)} missions. First and last evaluation per mission:")
        L += ["", "| seed | t | safe | dash | lean | choice | mission outcome |", "|---|---|---|---|---|---|---|"]
        for seed in fired:
            rows = [x for x in ov_rows if x[0] == seed]
            for _, o, outcome in ([rows[0], rows[-1]] if len(rows) > 1 else rows):
                p = o["plans"]
                L.append(f"| {seed} | {o['t']} | {p['safe']:.3f} | {p['dash']:.3f} | {p['lean']:.3f} | {o['choice']} | {outcome} |")
        risky = sum(1 for _, o, _ in ov_rows if o["choice"] != "safe")
        L.append("")
        L.append(f"{risky} of {len(ov_rows)} evaluations chose a risky plan.")
    L += ["", "## Failure causes", ""]
    L += [f"- {k}: {v}" for k, v in causes.most_common()] or ["- none"]
    L += ["", "## What this suggests we tune", ""]
    L += [f"- {s}" for s in _suggestions(runs, rate, life_frac or [0], causes, spend_mean, decoy_fooled, fooled, sweep)]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "n": n, "outcomes": oc, "rescue_rate": rate, "life_fraction_rescued": life_frac, "failure_causes": causes,
        "spend_mean": spend_mean, "readings": readings, "fooled": fooled,
        "cv": {k: dict(v) for k, v in cv.items()}}, indent=1, default=dict))
    return out / "summary.md"
