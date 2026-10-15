"""Command line: python -m signalsearch <command> ..."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _apply_sets(cfg: dict, sets: list[str] | None) -> dict:
    """--set world.target.k=1.6 style overrides (value parsed as YAML)."""
    import yaml
    for item in sets or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def cmd_run(a) -> None:
    from .common.config import load_config, resolve_config
    from .sim.runner import run_mission
    from .world.weather import resolve_weather
    cfg = resolve_weather(resolve_config(_apply_sets(load_config(a.config), a.set), a.seed))
    out = Path(a.out) if a.out else ROOT / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_seed{cfg['seed']}"
    summary = run_mission(cfg, out)
    print(json.dumps(summary))
    print(f"run saved to {out}")


def _run_job(job):
    from .sim.runner import run_mission
    cfg, out = job
    return run_mission(cfg, out)


def cmd_batch(a) -> None:
    from .common.config import load_config, resolve_config
    from .world.weather import resolve_weather
    base = _apply_sets(load_config(a.config), a.set)
    out = Path(a.out) if a.out else ROOT / "runs" / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    # fetch the weather once for the whole batch so every mission sees the same conditions
    weather = resolve_weather(resolve_config(base, 0))["world"]["weather"]["resolved"]
    for s in _seeds(a.seeds):
        cfg = resolve_config(copy.deepcopy(base), s)
        cfg["world"]["weather"]["resolved"] = dict(weather)
        jobs.append((cfg, out / f"seed{s:03d}"))
    rows = []
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(_run_job, jobs):
                rows.append(r)
                print(json.dumps(r), flush=True)
    else:
        for job in jobs:
            rows.append(_run_job(job))
            print(json.dumps(rows[-1]), flush=True)
    cfg_path = Path(a.config).resolve()
    cfg_name = str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else cfg_path.name
    (out / "batch.json").write_text(json.dumps({"config": cfg_name, "sets": a.set or [], "weather": weather,
                                                "runs": rows}, indent=1))
    ok = sum(r["outcome"] == "rescued" for r in rows)
    print(f"{ok}/{len(rows)} rescued; batch saved to {out}")


def cmd_analyze(a) -> None:
    from .analysis.analyze import analyze
    out = analyze(a.batch_dir, a.out or ROOT / "analysis", compare=a.compare)
    print(f"analysis written to {out}")


def cmd_diagnose(a) -> None:
    from .analysis.diagnose import analyse, diagnose_batch
    d = Path(a.run_dir)
    if (d / "trace.jsonl").exists():
        res = analyse(d)
        print(f"{d.name}: " + ", ".join(k for k, v in res["patterns"].items() if v) + f" -> {d / 'diagnosis.md'}")
    else:
        for res in diagnose_batch(d, a.jobs):
            print(f"{res['run']}: {res['outcome']}: " + (", ".join(k for k, v in res["patterns"].items() if v) or "clean"))


def cmd_replay(a) -> None:
    from .viz.replay import replay
    replay(a.run_dir, gif=a.gif)


def cmd_play(a) -> None:
    from .viz.play import play
    play(a.config, a.seed, a.as_)


def cmd_fetch_assets(a) -> None:
    from .viz.assets import fetch_kenney
    print(fetch_kenney(ROOT / "assets"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="signalsearch", description="WALL-E and EVE's Signal: autonomous search and rescue")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="headless mission -> runs/<timestamp>_seed<N>/")
    r.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    r.add_argument("--seed", type=int)
    r.add_argument("--out")
    r.add_argument("--set", action="append", help="override a config value, e.g. world.target.k=1.6")
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("batch", help="many missions")
    b.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    b.add_argument("--seeds", default="1-20")
    b.add_argument("--out")
    b.add_argument("--set", action="append", help="override a config value, e.g. world.target.k=1.6")
    b.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    b.set_defaults(fn=cmd_batch)
    an = sub.add_parser("analyze", help="charts + summary for a batch directory")
    an.add_argument("batch_dir")
    an.add_argument("--out")
    an.add_argument("--compare", nargs="*", help="other batch dirs (same seeds, one knob changed) for a sensitivity table")
    an.set_defaults(fn=cmd_analyze)
    dg = sub.add_parser("diagnose", help="whole-mission pattern diagnosis of a run (or every run in a batch)")
    dg.add_argument("run_dir")
    dg.add_argument("--jobs", type=int, default=1)
    dg.set_defaults(fn=cmd_diagnose)
    rp = sub.add_parser("replay", help="open the viewer on a saved run")
    rp.add_argument("run_dir")
    rp.add_argument("--gif", help="export an animated GIF instead of opening a window")
    rp.set_defaults(fn=cmd_replay)
    pl = sub.add_parser("play", help="human plays one side")
    pl.add_argument("--as", dest="as_", choices=["rescuer", "target"], required=True)
    pl.add_argument("--seed", type=int)
    pl.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    pl.set_defaults(fn=cmd_play)
    fa = sub.add_parser("fetch-assets", help="download the Kenney Tiny Dungeon pack (CC0)")
    fa.set_defaults(fn=cmd_fetch_assets)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
