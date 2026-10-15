"""The only place agent and world meet: agent.act(observation) -> action, env.step(action) -> observation."""
from __future__ import annotations

import time
from pathlib import Path

from ..agent.agent import Agent
from ..agent.report import to_markdown
from ..common.config import load_config, resolve_config, save_config
from ..world.env import Env
from ..world.weather import resolve_weather
from .logger import RunLogger


def prepare_config(config_path: str | Path, seed: int | None = None, log=print) -> dict:
    cfg = load_config(config_path)
    return resolve_weather(resolve_config(cfg, seed), log=log)


def make_agent(cfg: dict) -> Agent:
    # the agent gets its policy knobs and the agent_assumed section; nothing from `world`
    return Agent(cfg["agent"], cfg["agent_assumed"], seed=cfg["seeds"]["agent"])


def run_mission(cfg: dict, out_dir: str | Path | None = None, on_turn=None, target_input=None,
                quiet: bool = True) -> dict:
    """Run one mission. `on_turn(env, agent, obs)` is a viewer hook; `target_input()` returns a
    human-chosen target move (true coordinates) or None, enabling play --as target."""
    env = Env(cfg)
    agent = make_agent(cfg)
    log = RunLogger(out_dir) if out_dir else None
    if log:
        save_config(cfg, Path(out_dir) / "config.resolved.yaml")
        log.world(env)
    obs = env.reset()
    t0 = time.time()
    while True:
        action = agent.act(obs)
        if log:
            log.turn(obs, agent.decision if action is not None else {"action": None, "reason": "episode over"}, agent)
            log.truth_line(env.truth_snapshot())
        if on_turn:
            on_turn(env, agent, obs)
        if action is None:
            break
        if target_input is not None:
            obs = env.step(action, target_move=target_input(env), target_controlled=True)
        else:
            obs = env.step(action)
    rep = agent.final_report()
    if log:
        log.report(rep, to_markdown(rep))
        log.close()
        from .evaluate import evaluate
        evaluate(out_dir)
    summary = {"seed": cfg["seed"], "outcome": env.reason, "turns": env.t, "life_turns": env.life["life_turns"],
               "battery": env.battery, "seconds": round(time.time() - t0, 1)}
    if not quiet:
        print(summary)
    return summary
