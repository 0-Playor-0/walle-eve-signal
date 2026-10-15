"""Evaluator: compares a finished run with ground truth. Runs after the agent's report is written and
appends a "Ground truth comparison" section; also writes eval.json for the batch analysis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .logger import _json_default, load_jsonl

KINDS = ("target", "dog", "torch", "hearth", "armour")


def _entities_at(world: dict, truth_t: dict) -> list[tuple[str, list[int]]]:
    out = [("target", truth_t["target"])]
    dogs = {d[0]: d[1:] for d in truth_t["dogs"]}
    for e in world["entities"]:
        if e["kind"] == "target":
            continue
        out.append((e["kind"], dogs.get(e["id"], e["pos"]) if e["kind"] == "dog" else e["pos"]))
    return out


def _attribute(pos_true, ents, gate: float):
    best, bd = None, gate
    for kind, p in ents:
        d = math.dist(pos_true, p)
        if d <= bd:
            best, bd = kind, d
    return best or "none"


def evaluate(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    world = json.loads((run / "world.json").read_text())
    truth = load_jsonl(run / "truth.jsonl")
    trace = load_jsonl(run / "trace.jsonl")
    rep = json.loads((run / "report.json").read_text())
    snaps = np.load(run / "beliefs.npz")
    ox, oy = world["offset"]
    to_agent = lambda p: [p[0] + ox, p[1] + oy]
    to_true = lambda p: [p[0] - ox, p[1] - oy]
    truth_by_t = {tr["t"]: tr for tr in truth}

    # belief mass within 2 cells of the true target, every turn
    mass_at_truth = []
    for tr in truth:
        t = tr["t"]
        key = f"b{t:05d}"
        if key not in snaps:
            continue
        x0, x1, y0, y1 = snaps[f"w{t:05d}"].tolist()
        b = snaps[key].astype(np.float64)
        ax, ay = to_agent(tr["target"])
        cx, cy = ax - x0, ay - y0
        m = float(b[max(0, cx - 2):cx + 3, max(0, cy - 2):cy + 3].sum()) if 0 <= cx < b.shape[0] and 0 <= cy < b.shape[1] else 0.0
        mass_at_truth.append([t, round(m / max(b.sum(), 1e-12), 4)])

    # what each reading was really caused by
    by_sensor = {s: {k: 0 for k in KINDS + ("none",)} for s in ("thermal", "radar", "lidar_humanoid", "cv_target")}
    cv_labels = {k: {} for k in KINDS}
    fooled = {s: {k: 0 for k in KINDS} for s in ("thermal", "radar", "lidar_humanoid")}
    peaks = {rec["t"]: rec.get("belief", {}).get("peak") for rec in trace}
    for rec in trace:
        t = rec["t"]
        obs = rec["observation"]
        # readings at turn t were taken before the world ticked: use the previous truth snapshot
        tt = truth_by_t.get(t - 1, truth_by_t.get(t))
        if tt is None:
            continue
        ents = _entities_at(world, tt)
        r = obs.get("readings", {})
        items = []
        for b in r.get("thermal", []):
            items.append(("thermal", b["pos"], b["radius"] + 2))
        for b in r.get("radar", []):
            items.append(("radar", b["pos"], 3.5))
        for b in r.get("lidar", {}).get("blobs", []):
            if b["shape"] == "humanoid":
                items.append(("lidar_humanoid", b["pos"], 0.5))
        for lab in r.get("cv", {}).get("labels", []):
            kind = _attribute(to_true(lab["pos"]), ents, 0.5)
            if kind in cv_labels:
                cv_labels[kind][lab["label"]] = cv_labels[kind].get(lab["label"], 0) + 1
            if lab["label"] == "target":
                items.append(("cv_target", lab["pos"], 0.5))
        for sensor, pos, gate in items:
            kind = _attribute(to_true(pos), ents, gate)
            by_sensor[sensor][kind] += 1
            if sensor in fooled and kind not in ("target", "none"):
                # fooled = within the next 5 turns the belief peak sits on that decoy
                src = [p for k, p in ents if k == kind]
                for dt in range(1, 6):
                    pk = peaks.get(t + dt)
                    if pk and any(math.dist(to_true(pk), p) <= 2 for p in src):
                        fooled[sensor][kind] += 1
                        break

    last = truth[-1]
    final_true = to_agent(last["target"])
    inferred = rep["inferred_location"]["agent_frame"]
    err = abs(final_true[0] - inferred[0]) + abs(final_true[1] - inferred[1])
    outcome = rep["outcome"]
    final_mass = mass_at_truth[-1][1] if mass_at_truth else 0.0
    max_mass = max((m for _, m in mass_at_truth), default=0.0)
    if outcome == "rescued":
        cause = None
    elif outcome == "battery_depleted":
        cause = "battery depleted"
    elif max_mass < 0.1:
        cause = "never localized the target"
    elif not any(rec["observation"].get("readings", {}).get("cv", {}).get("labels") and
                 any(l["label"] == "target" for l in rec["observation"]["readings"]["cv"]["labels"]) for rec in trace):
        cause = "localized but never CV-confirmed"
    else:
        cause = "confirmed but could not reach the target in time"
    ev = {
        "outcome": outcome, "turns": rep["turns"], "life_turns": world["life"]["life_turns"],
        "d_path": world["life"]["d_path"], "turn_fraction_of_life": round(rep["turns"] / world["life"]["life_turns"], 3),
        "final_true_target_agent_frame": final_true, "inferred_agent_frame": inferred, "location_error_manhattan": err,
        "true_energy_final": last["energy"], "battery": rep["battery"], "spend": rep["spend"],
        "recharges": rep["recharges"], "overrides": rep["overrides"], "mass_at_truth": mass_at_truth,
        "max_mass_at_truth": max_mass, "final_mass_at_truth": final_mass, "readings_by_source": by_sensor,
        "cv_labels_by_true_kind": cv_labels, "fooled": fooled, "failure_cause": cause,
        "true_life_remaining_final": last["remaining_life"],
        "estimated_life_remaining_final": rep["remaining_life_estimate"]["mean"],
    }
    with open(run / "eval.json", "w") as f:
        json.dump(ev, f, default=_json_default, indent=1)
    md = ["", "## Ground truth comparison", "", "_Appended by the evaluator; the agent never saw any of this._", "",
          f"- True target position (final): agent frame {final_true}, map {last['target']}",
          f"- Agent's inferred position: {inferred} -> error {err} cells (Manhattan)",
          f"- True energy at the end: {last['energy']:.3f}; true remaining life {last['remaining_life']} turns "
          f"vs estimated {rep['remaining_life_estimate']['mean']}",
          f"- Mission life budget: {world['life']['life_turns']} turns (walking distance {world['life']['d_path']}); "
          f"used {rep['turns']} turns",
          f"- Belief mass within 2 cells of the truth: final {final_mass:.3f}, best {max_mass:.3f}",
          f"- Failure cause: {cause or 'n/a (rescued)'}", "",
          "| sensor | " + " | ".join(KINDS + ("none",)) + " |", "|---" * (len(KINDS) + 2) + "|"]
    for s, row in by_sensor.items():
        md.append(f"| {s} | " + " | ".join(str(row[k]) for k in KINDS + ("none",)) + " |")
    md += ["", f"CV labels by true entity: {cv_labels}"]
    with open(run / "report.md", "a") as f:
        f.write("\n".join(md) + "\n")
    return ev
