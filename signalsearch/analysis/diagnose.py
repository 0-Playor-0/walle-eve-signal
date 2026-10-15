"""Whole-mission diagnostics for one run (evaluator tool: may read ground truth).

Reads every turn of trace.jsonl + truth.jsonl and finds *sequences* that totals hide: charging loops,
oscillation, plan churn, idle streaks, "near but not closing", CV attempts that could not see the target,
confirm sequences that stall. Writes diagnosis.md + diagnosis.png into the run directory.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..common.actions import DIRS  # noqa: E402
from ..sim.logger import load_jsonl  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ACTION_COL = {"move": SERIES[0], "turn": SERIES[6], "lidar": SERIES[2], "radar": SERIES[3], "thermal": SERIES[4],
              "cv": SERIES[1], "charge": SERIES[5], "wait": SERIES[7]}

NEAR = 6            # "near" the target (cells, Euclidean)
NEAR_TURNS = 10     # ... for this long without a CV attempt
SHORT_EXCURSION = 20
OSC_WINDOW, OSC_CELLS = 20, 4
CHURN_WINDOW, CHURN_RESETS = 20, 5
IDLE_MIN = 5


def _ranges(ts: list[int], gap: int = 1) -> list[tuple[int, int]]:
    out = []
    for t in sorted(ts):
        if out and t - out[-1][1] <= gap:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def _fmt(rs) -> str:
    if not rs:
        return "-"
    s = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in rs[:12])
    return s + (f" (+{len(rs) - 12} more)" if len(rs) > 12 else "")


def cv_pattern(pose, facing):
    dx, dy = DIRS[facing]
    lx, ly = -dy, dx
    return [(pose[0] + dx, pose[1] + dy)] + [(pose[0] + 2 * dx + k * lx, pose[1] + 2 * dy + k * ly) for k in (-1, 0, 1)]


def confirm_step(reason: str) -> str | None:
    """Which step of the confirm sequence a turn belongs to (from the logged reason)."""
    r = reason or ""
    if not (r.startswith("candidate at") or r.startswith("source ") or "[dash override] candidate" in r
            or "[lean override] candidate" in r or "confirm:" in r):
        return None
    for key, step in (("LIDAR first", "lidar"), ("radar first", "radar"), ("approach", "approach"),
                      ("turn to face", "face"), ("CV to confirm", "cv"), ("step off", "step-off"),
                      ("new viewpoint", "approach"), ("door", "approach")):
        if key in r:
            return step
    return "other"


class Run:
    def __init__(self, run_dir):
        d = Path(run_dir)
        self.dir = d
        self.trace = load_jsonl(d / "trace.jsonl")
        self.truth = {r["t"]: r for r in load_jsonl(d / "truth.jsonl")}
        self.world = json.loads((d / "world.json").read_text())
        self.walls = np.load(d / "world.npz")["walls"]
        self.eval = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
        ox, oy = self.world["offset"]
        self.off = (ox, oy)
        self.mass = dict((t, m) for t, m in self.eval.get("mass_at_truth", []))

    def tgt(self, t):
        """True target position in the agent frame at the *end* of turn t."""
        tr = self.truth.get(t)
        return None if tr is None else (tr["target"][0] + self.off[0], tr["target"][1] + self.off[1])

    def wall_agent(self, c) -> bool:
        x, y = c[0] - self.off[0], c[1] - self.off[1]
        n = self.walls.shape[0]
        return not (0 <= x < n and 0 <= y < n) or bool(self.walls[x, y])


def analyse(run_dir) -> dict:
    R = Run(run_dir)
    tr = R.trace
    T = [r["t"] for r in tr]
    obs = [r["observation"] for r in tr]
    acts = [(r["observation"]["action"] or "").split()[0] for r in tr]  # action executed at turn t
    pose = [tuple(o["pose"]) for o in obs]
    dist = [math.dist(pose[i], R.tgt(T[i]) or pose[i]) for i in range(len(tr))]
    first_loc = next((t for t in T if R.mass.get(t, 0) >= 0.5), None)
    first_10 = next((t for t in T if R.mass.get(t, 0) >= 0.1), None)

    # ---- hearth excursions and charging loops ----------------------------------------------------------
    excursions = []  # (leave_t, return_t or None)
    start = None
    for i in range(1, len(tr)):
        if obs[i - 1]["on_hearth"] and not obs[i]["on_hearth"]:
            start = T[i]
        elif not obs[i - 1]["on_hearth"] and obs[i]["on_hearth"] and start is not None:
            excursions.append((start, T[i]))
            start = None
    if start is not None:
        excursions.append((start, None))
    short = [e for e in excursions if e[1] is not None and e[1] - e[0] < SHORT_EXCURSION]
    loops = []  # runs of >= 2 consecutive short excursions
    run = []
    for e in excursions:
        if e[1] is not None and e[1] - e[0] < SHORT_EXCURSION:
            run.append(e)
        else:
            if len(run) >= 2:
                loops.append((run[0][0], run[-1][1], len(run)))
            run = []
    if len(run) >= 2:
        loops.append((run[0][0], run[-1][1], len(run)))
    charges = [T[i] for i in range(len(tr)) if acts[i] == "charge"]

    # ---- oscillation ------------------------------------------------------------------------------------
    osc = []
    for i in range(OSC_WINDOW, len(tr)):
        w = pose[i - OSC_WINDOW:i]
        moves = sum(1 for a in acts[i - OSC_WINDOW:i] if a == "move")
        if moves >= OSC_WINDOW // 2 and len(set(w)) <= OSC_CELLS:
            osc.append(T[i])
    osc_r = _ranges(osc, 2)

    # ---- plan churn ---------------------------------------------------------------------------------------
    goals = [tuple((r.get("plan") or {}).get("route_goal") or (r.get("plan") or {}).get("goal") or ()) for r in tr]
    resets = [T[i] for i in range(1, len(tr)) if goals[i] and goals[i - 1] and goals[i] != goals[i - 1]]
    churn = [t for t in resets if sum(1 for u in resets if t - CHURN_WINDOW < u <= t) >= CHURN_RESETS]
    churn_r = _ranges(churn, CHURN_WINDOW // 2)

    # ---- idle streaks ---------------------------------------------------------------------------------------
    idle_t = []
    for i in range(1, len(tr)):
        a = acts[i]
        same = pose[i] == pose[i - 1]
        if a == "wait" or (a == "move" and obs[i]["blocked"]) or (a == "charge" and not obs[i].get("charged")):
            idle_t.append(T[i])
        elif a == "turn" and same and (i + 1 >= len(tr) or acts[i + 1] not in ("cv", "turn")):
            idle_t.append(T[i])
    idle_r = [r for r in _ranges(idle_t) if r[1] - r[0] + 1 >= IDLE_MIN]

    # ---- near but not closing ------------------------------------------------------------------------------
    near_t = [T[i] for i in range(len(tr)) if dist[i] <= NEAR]
    nbnc = []
    for a, b in _ranges(near_t):
        if b - a + 1 >= NEAR_TURNS and not any(acts[i] == "cv" for i in range(len(tr)) if a <= T[i] <= b):
            nbnc.append((a, b))

    # ---- CV attempts -------------------------------------------------------------------------------------------
    cv_rows = []
    for i in range(len(tr)):
        if acts[i] != "cv":
            continue
        o = obs[i]
        t = T[i]
        tg = R.tgt(t - 1)  # readings are taken before the world ticks
        pat = cv_pattern(o["pose"], o["facing"])
        C = o["readings"].get("cv", {})
        labels = [lab["label"] for lab in C.get("labels", [])]
        saw_target = "target" in labels
        if saw_target:
            verdict = "saw target"
        elif not C.get("visible"):
            verdict = "only a wall (front cell is wall)"
        elif tg in pat and tuple(tg) in {tuple(c) for c in C.get("visible", [])}:
            verdict = "target in view but missed (5% miss)"
        elif tg in pat:
            verdict = "target in pattern but line of sight blocked"
        elif math.dist(o["pose"], tg) <= 3:
            prev = R.tgt(t - 2)
            verdict = "target moved out of pattern" if prev in pat else "target adjacent but outside pattern"
        else:
            verdict = f"target far ({math.dist(o['pose'], tg):.0f} cells)"
        cv_rows.append({"t": t, "pose": o["pose"], "facing": o["facing"], "labels": labels, "verdict": verdict,
                        "seen": len(C.get("visible", [])), "walls": len(C.get("walls", []))})
    repeat_wall_cv = 0
    seen_wall = set()
    for c in cv_rows:
        key = (tuple(c["pose"]), c["facing"])
        if c["seen"] == 0:
            repeat_wall_cv += key in seen_wall
            seen_wall.add(key)

    # ---- confirm sequences ----------------------------------------------------------------------------------------
    steps = [confirm_step(r.get("reason")) for r in tr]
    seqs = []
    i = 0
    while i < len(tr):
        if steps[i]:
            j = i
            while j + 1 < len(tr) and (steps[j + 1] or (j + 2 < len(tr) and steps[j + 2])):
                j += 1
            ss = [s for s in steps[i:j + 1] if s]
            reached_cv = "cv" in ss
            stall = None if reached_cv else Counter(ss).most_common(1)[0][0]
            seqs.append({"start": T[i], "end": T[j], "steps": dict(Counter(ss)), "reached_cv": reached_cv, "stall": stall})
            i = j + 1
        else:
            i += 1
    stalled = [s for s in seqs if not s["reached_cv"] and s["end"] - s["start"] >= 5]

    # ---- near-failing safety and modes -----------------------------------------------------------------------------
    near_fail = [r["t"] for r in tr if (r.get("safety") or {}).get("near_failing")]
    modes = [r.get("mode") or "end" for r in tr]
    phases = []
    for i, m in enumerate(modes):
        if phases and phases[-1][0] == m:
            phases[-1][2] = T[i]
        else:
            phases.append([m, T[i], T[i]])
    hist_all = Counter(acts[1:])
    hist_after = Counter(acts[i] for i in range(1, len(tr)) if first_loc is not None and T[i] > first_loc)
    overrides = [r["override"] for r in tr if r.get("override")]
    escalations = [r["t"] for r in tr if r.get("escalation")]
    res = {
        "run": R.dir.name, "outcome": R.eval.get("outcome"), "turns": T[-1], "life": R.world["life"]["life_turns"],
        "battery_end": obs[-1]["battery"], "first_10": first_10, "first_loc": first_loc,
        "excursions": excursions, "short_excursions": len(short), "charging_loops": loops, "charges": charges,
        "oscillation": osc_r, "plan_resets": len(resets), "churn": churn_r, "idle": idle_r,
        "near_not_closing": nbnc, "cv": cv_rows, "repeat_wall_cv": repeat_wall_cv, "confirm_sequences": seqs,
        "confirm_stalled": stalled, "near_failing": near_fail, "phases": phases, "hist_all": dict(hist_all),
        "hist_after": dict(hist_after), "overrides": len(overrides), "escalations": escalations,
        "min_dist": min(dist), "turns_near": len(near_t),
        "patterns": {
            "charging_loop": len(loops) > 0, "oscillation": len(osc_r) > 0, "plan_churn": len(churn_r) > 0,
            "idle_streak": len(idle_r) > 0, "near_not_closing": len(nbnc) > 0,
            "blocked_cv": any(c["verdict"] not in ("saw target",) and not c["verdict"].startswith("target far") for c in cv_rows),
            "confirm_stall": len(stalled) > 0,
            "no_cv_after_localization": R.eval.get("outcome") != "rescued" and first_loc is not None
            and not any(c["t"] >= first_loc for c in cv_rows),
        },
    }
    _chart(R, T, acts, dist, obs, charges, near_fail, first_loc, R.dir / "diagnosis.png")
    (R.dir / "diagnosis.md").write_text(_markdown(res))
    (R.dir / "diagnosis.json").write_text(json.dumps(res, indent=1, default=list))
    return res


def _narrative(res) -> list[str]:
    out = []
    loc = res["first_loc"]
    ph = res["phases"]
    # compress modes into the story: search runs and the hearth cycle
    cur = []
    for m, a, b in ph:
        out_m = {"charging": "charge", "topup": "top-up detour", "return": "return to hearth"}.get(m, m)
        cur.append(f"t={a}-{b} {out_m}" if a != b else f"t={a} {out_m}")
    if loc is not None:
        out.append(f"- t={loc}: **localized** (belief >= 50% within 2 cells of the truth)")
    elif res["first_10"] is not None:
        out.append(f"- t={res['first_10']}: belief first puts 10% near the truth (never 50%)")
    else:
        out.append("- never localized (belief never put 10% within 2 cells of the truth)")
    if res["charging_loops"]:
        for a, b, n in res["charging_loops"]:
            out.append(f"- t={a}-{b}: **charging loop**, {n} consecutive excursions under {SHORT_EXCURSION} turns")
    if res["near_not_closing"]:
        out.append(f"- near (<= {NEAR} cells) but no CV for 10+ turns: t={_fmt(res['near_not_closing'])}")
    for c in res["cv"]:
        out.append(f"- t={c['t']}: CV from {c['pose']} facing {c['facing']} -> {c['labels'] or 'nothing'} ({c['verdict']})")
    out.append(f"- end t={res['turns']}: {res['outcome']}, battery {res['battery_end']:.1f}%")
    if len(cur) > 40:
        cur = cur[:20] + [f"... {len(cur) - 40} more mode switches ..."] + cur[-20:]
    return out, cur


def _markdown(res) -> str:
    story, modes = _narrative(res)
    p = res["patterns"]
    L = [f"# Diagnosis: {res['run']}", "",
         f"Outcome **{res['outcome']}** after {res['turns']} of {res['life']} life turns; battery at the end "
         f"{res['battery_end']:.1f}%. Closest approach to the target: {res['min_dist']:.1f} cells "
         f"({res['turns_near']} turns within {NEAR}).", "", "![timeline](diagnosis.png)", "",
         "## Timeline", ""] + story + ["", "Mode sequence: " + "; ".join(modes), "",
         "## Pattern detectors", "", "| pattern | found | evidence |", "|---|---|---|",
         f"| charging loop (>= 2 consecutive excursions < {SHORT_EXCURSION} turns) | {p['charging_loop']} | "
         + (", ".join(f"t={a}-{b} ({n} trips)" for a, b, n in res['charging_loops']) or "-") + " |",
         f"| oscillation (<= {OSC_CELLS} cells in {OSC_WINDOW} turns) | {p['oscillation']} | t={_fmt(res['oscillation'])} |",
         f"| plan churn (>= {CHURN_RESETS} goal changes in {CHURN_WINDOW} turns) | {p['plan_churn']} | "
         f"t={_fmt(res['churn'])} ({res['plan_resets']} goal changes total) |",
         f"| idle streaks (>= {IDLE_MIN} zero-progress turns) | {p['idle_streak']} | t={_fmt(res['idle'])} |",
         f"| near but not closing | {p['near_not_closing']} | t={_fmt(res['near_not_closing'])} |",
         f"| CV that could not see the target | {p['blocked_cv']} | "
         + ("; ".join(f"t={c['t']} {c['verdict']}" for c in res['cv'] if c['verdict'] != 'saw target') or "-") + " |",
         f"| confirm sequence stalled before CV | {p['confirm_stall']} | "
         + ("; ".join(f"t={s['start']}-{s['end']} stuck on {s['stall']}" for s in res['confirm_stalled']) or "-") + " |",
         f"| no CV at all after localization | {p['no_cv_after_localization']} | |", "",
         "## Actions", "", "| action | whole mission | after localization |", "|---|---|---|"]
    for a in sorted(set(res["hist_all"]) | set(res["hist_after"])):
        L.append(f"| {a} | {res['hist_all'].get(a, 0)} | {res['hist_after'].get(a, 0)} |")
    L += ["", f"Hearth excursions: {len(res['excursions'])} ({res['short_excursions']} under {SHORT_EXCURSION} turns); "
          f"charge actions: {len(res['charges'])}; override evaluations: {res['overrides']}; "
          f"escalations: {len(res['escalations'])}; near-failing safety checks: {len(res['near_failing'])}.", ""]
    return "\n".join(L) + "\n"


def _chart(R, T, acts, dist, obs, charges, near_fail, first_loc, path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 0.7]})
    fig.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        if first_loc is not None:
            ax.axvline(first_loc, color=INK2, linewidth=1, linestyle="--")
    axes[0].plot(T, dist, color=SERIES[0], linewidth=1.6)
    axes[0].axhline(NEAR, color=GRID, linewidth=1)
    axes[0].set_ylabel("true distance\nto target", color=INK2, fontsize=8)
    axes[0].set_title(f"{R.dir.name}: {R.eval.get('outcome')} at t={T[-1]}" +
                      (f"  (dashed: localized at t={first_loc})" if first_loc is not None else ""),
                      loc="left", color=INK, fontsize=11, fontweight="bold")
    mt = sorted(R.mass.items())
    axes[1].plot([t for t, _ in mt], [m for _, m in mt], color=SERIES[1], linewidth=1.6)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("belief mass\nnear truth", color=INK2, fontsize=8)
    axes[2].plot(T, [o["battery"] for o in obs], color=SERIES[2], linewidth=1.6)
    if charges:
        axes[2].scatter(charges, [obs[T.index(t)]["battery"] for t in charges], s=14, color=SERIES[5], zorder=3,
                        label="charge")
    if near_fail:
        axes[2].scatter(near_fail, [obs[T.index(t)]["battery"] for t in near_fail], s=10, marker="x",
                        color=SERIES[7], zorder=3, label="near-failing safety check")
    if charges or near_fail:
        axes[2].legend(frameon=False, fontsize=7, loc="upper right")
    axes[2].set_ylim(0, 102)
    axes[2].set_ylabel("battery %", color=INK2, fontsize=8)
    for a, col in ACTION_COL.items():
        ts = [T[i] for i in range(len(T)) if acts[i] == a]
        axes[3].scatter(ts, [0] * len(ts), marker="|", s=80, color=col, label=a)
    axes[3].set_yticks([])
    axes[3].legend(frameon=False, fontsize=7, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.6))
    axes[3].set_xlabel("turn", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def diagnose_batch(batch_dir, jobs: int = 1) -> list[dict]:
    dirs = [d for d in sorted(Path(batch_dir).glob("seed*")) if (d / "trace.jsonl").exists()]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(analyse, dirs))
    return [analyse(d) for d in dirs]


PATTERNS = ["charging_loop", "oscillation", "plan_churn", "idle_streak", "near_not_closing", "blocked_cv",
            "confirm_stall", "no_cv_after_localization"]


def pattern_counts(results: list[dict]) -> dict:
    """Missions showing each pattern, plus turn totals for the ones that waste time."""
    out = {p: sum(1 for r in results if r["patterns"].get(p)) for p in PATTERNS}
    out["charging_loop_trips"] = sum(n for r in results for _, _, n in r["charging_loops"])
    out["idle_turns"] = sum(b - a + 1 for r in results for a, b in r["idle"])
    out["near_not_closing_turns"] = sum(b - a + 1 for r in results for a, b in r["near_not_closing"])
    out["stalled_approach_turns"] = sum(s["end"] - s["start"] + 1 for r in results for s in r["confirm_stalled"])
    out["cv_not_seeing_target"] = sum(1 for r in results for c in r["cv"]
                                      if c["verdict"] != "saw target" and not c["verdict"].startswith("target far"))
    out["repeat_wall_cv"] = sum(r["repeat_wall_cv"] for r in results)
    return out
