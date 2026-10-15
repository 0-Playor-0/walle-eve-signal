"""Final mission report, written from agent state only."""
from __future__ import annotations


def _status(agent) -> tuple[str, dict]:
    life = agent.life.remaining(agent.t)
    e = agent.motion.energy_estimate(agent.t)
    if agent.outcome == "rescued":
        return "alive (rescued)", life
    if agent.outcome == "target_died":
        return "deceased (environment reported the episode over: target_died)", life
    word = "alive" if e > 0.4 else "alive but weak" if e > 0.1 else "critical"
    return f"{word} (estimated energy {e:.2f})", life


def _recommend(agent, peak, mass) -> str:
    hearth = agent.nearest_hearth()
    if agent.outcome == "rescued":
        where = f"hearth at {list(hearth)} ({agent.h} moves)" if hearth else "the spawn hearth"
        return f"Escort the target to the {where}, keep them warm, and call for medical support."
    if agent.outcome == "target_died":
        return (f"Mission failed. Recover the target at the most likely cell {list(peak)} (P={mass:.2f}); "
                f"start the next mission sooner or with more battery.")
    step = f"Resume the search at {list(peak)} (P={mass:.2f} within 2 cells): approach, face it, CV to confirm"
    if agent.battery < 40 and hearth:
        step = f"Recharge at {list(hearth)} first, then r" + step[1:]
    return step + "."


def build(agent) -> dict:
    peaks = agent.belief.peaks(3)
    peak, mass = peaks[0] if peaks else (tuple(agent.pose), 0.0)
    status, life = _status(agent)
    evidence = []
    for tr in agent.tracker.tracks:
        for (t, x, y, sigma, sensor) in tr.history:
            evidence.append({"t": t, "sensor": sensor, "pos": [round(x, 1), round(y, 1)], "sigma": round(sigma, 2),
                             "track": tr.id, "interpretation": tr.guess()})
    evidence.sort(key=lambda e: (e["t"], e["track"]))
    confirmed = agent.last_cv_target is not None
    confidence = min(1.0, mass) if not agent.outcome == "rescued" else 1.0
    rep = {
        "outcome": agent.outcome,
        "turns": agent.t,
        "battery": round(agent.battery, 2),
        "inferred_location": {"agent_frame": list(peak), "map": agent.frame.to_map(peak),
                              "mass_within_2": round(mass, 4),
                              "alternatives": [{"agent_frame": list(c), "mass_within_2": round(m, 4)} for c, m in peaks[1:]],
                              "offset_resolved": agent.frame.resolved, "feasible_window": list(agent.frame.window)},
        "status": status,
        "remaining_life_estimate": life,
        "p_move_estimate": round(agent.motion.estimate(agent.t), 4),
        "confidence": round(confidence, 3),
        "cv_confirmed": confirmed,
        "recommended_next_step": _recommend(agent, peak, mass),
        "tracks": [tr.to_dict() for tr in agent.tracker.tracks],
        "evidence": evidence,
        "overrides": agent.overrides,
        "spend": {k: round(v, 2) for k, v in agent.spend.items()},
        "recharges": agent.recharges,
    }
    return rep


def to_markdown(rep: dict) -> str:
    L = rep["inferred_location"]
    lines = [
        "# Mission report", "",
        f"- **Outcome:** {rep['outcome']} after {rep['turns']} turns",
        f"- **Battery left:** {rep['battery']}%  (recharges: {rep['recharges']})",
        f"- **Inferred location:** agent frame {L['agent_frame']}"
        + (f", map {L['map']}" if L["map"] else " (map offset not fully resolved)")
        + f" - belief mass within 2 cells {L['mass_within_2']}",
        f"- **Status:** {rep['status']}",
        f"- **Estimated remaining life:** mean {rep['remaining_life_estimate']['mean']} turns "
        f"(10-90%: {rep['remaining_life_estimate']['p10']}-{rep['remaining_life_estimate']['p90']})",
        f"- **Confidence:** {rep['confidence']}  (CV confirmed: {rep['cv_confirmed']})",
        f"- **Recommended next step:** {rep['recommended_next_step']}", "",
        "## Sources tracked", "",
        "| track | position | hits | CV labels | interpretation | speed |", "|---|---|---|---|---|---|",
    ]
    for tr in rep["tracks"]:
        lines.append(f"| {tr['id']} | {tr['pos']} | {tr['hits']} | {tr['labels'] or '-'} | {tr['guess']} | {tr['speed']} |")
    lines += ["", "## Evidence log", "", "| turn | sensor | position | track | track's interpretation (at the end) |", "|---|---|---|---|---|"]
    for e in rep["evidence"][:200]:
        lines.append(f"| {e['t']} | {e['sensor']} | {e['pos']} | {e['track']} | {e['interpretation']} |")
    if len(rep["evidence"]) > 200:
        lines.append(f"| ... | {len(rep['evidence']) - 200} more in report.json | | | |")
    lines += ["", "## Battery spend", ""]
    for k, v in rep["spend"].items():
        lines.append(f"- {k}: {v}%")
    lines += ["", "## Endgame overrides", ""]
    if not rep["overrides"]:
        lines.append("None fired.")
    for o in rep["overrides"][:30]:
        lines.append(f"- t={o['t']}: scores {o['plans']} -> **{o['choice']}**")
    if len(rep["overrides"]) > 30:
        lines.append(f"- ... {len(rep['overrides']) - 30} more in report.json")
    return "\n".join(lines) + "\n"
