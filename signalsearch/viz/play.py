"""Human-controlled side.

--as target  : you move the princess (energy rules still apply); the agent hunts you.
--as rescuer : you choose the knight's actions; you see only the mind view (the same occupancy + belief
               the agent would build from your observations) until the mission ends.
"""
from __future__ import annotations

import pygame

from ..common.actions import CHARGE, CV, DIRS, LIDAR, MOVE, RADAR, THERMAL, TURN, WAIT, Action
from ..sim.runner import make_agent, prepare_config
from ..world.env import Env
from .viewer import H, W, Viewer, events_of, handle_common_keys, receipt, weather_str

ARROWS = {pygame.K_UP: "N", pygame.K_RIGHT: "E", pygame.K_DOWN: "S", pygame.K_LEFT: "W",
          pygame.K_w: "N", pygame.K_d: "E", pygame.K_s: "S", pygame.K_a: "W"}
RESCUER_KEYS = {pygame.K_l: Action(LIDAR), pygame.K_r: Action(RADAR), pygame.K_t: Action(THERMAL),
                pygame.K_c: Action(CV), pygame.K_h: Action(CHARGE), pygame.K_SPACE: Action(WAIT),
                pygame.K_q: Action(TURN, "L"), pygame.K_e: Action(TURN, "R")}
SENSORS = ("lidar", "radar", "thermal", "cv")


def live_frame(env: Env, agent, obs, trail: list, weather: str) -> dict:
    x0, x1, y0, y1 = agent.frame.window
    act = obs.action.split()[0]
    route = agent.route
    path = None
    if route:
        i = route["path"].index(agent.pose) + 1 if agent.pose in route["path"] else 0
        path = [list(c) for c in route["path"][i:i + 60]]
    d = agent.decision or {}
    return {
        "t": obs.t, "battery": obs.battery, "reason": d.get("reason", ""), "mode": d.get("mode", agent.mode),
        "override": d.get("override"), "outcome": obs.episode["reason"] if obs.done else None, "weather": weather,
        "mind": {"window": [x0, x1, y0, y1], "occ": agent.occ.grid[x0:x1 + 1, y0:y1 + 1],
                 "belief": agent.belief.p[x0:x1 + 1, y0:y1 + 1], "pose": list(agent.pose), "facing": agent.facing,
                 "hearths": [list(h) for h in agent.occ.hearths], "path": path,
                 "goal": list(route["goal"]) if route else None,
                 "scan": {"kind": act, "pose": obs.pose, "facing": obs.facing} if act in SENSORS else None,
                 "fused": agent.fused, "life": agent.life.remaining(obs.t), "p_move": agent.motion.estimate(obs.t),
                 "readings": obs.readings or {}},
        "truth": env.truth_snapshot(), "trail": trail,
    }


def play(config, seed, side: str) -> None:
    cfg = prepare_config(config, seed)
    env = Env(cfg)
    agent = make_agent(cfg)
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption(f"Play as {side} - seed {cfg['seed']}")
    viewer = Viewer(cfg, env.map.walls, env.static_truth())
    weather = weather_str(env.weather)
    clock = pygame.time.Clock()
    obs = env.reset()
    receipts = [receipt(obs.to_dict())]
    trail = [list(env.pos)]
    if side == "rescuer":
        agent.observe(obs)            # the "mind" is built from your observations
        action = None
        agent.decision = {"reason": "your move: arrows move, Q/E turn, L lidar, R radar, T thermal, C cv, H charge, space wait",
                          "mode": "you"}
    else:
        action = agent.act(obs)
    pending = None
    tps = float(cfg.get("viz", {}).get("turns_per_second", 8)) / 2
    acc, tick, paused, running = 0.0, 0, False, True
    help_r = "arrows move  Q/E turn  L lidar  R radar  T thermal  C cv  H charge  space wait  B/F/P/M layers  G legend  Esc quit"
    help_t = "arrows/WASD move the target (else she stays)  P pause  +/- speed  B/F/M layers  L legend  Esc quit"
    viewer.add_events(0, [])
    while running:
        dt = clock.tick(30) / 1000.0
        tick += 1
        step_now = None
        for ev in pygame.event.get():  # every frame, or macOS never shows the window
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif obs.done:
                    continue
                elif side == "rescuer":
                    if ev.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT):
                        step_now = Action(MOVE, ARROWS[ev.key])
                    elif ev.key in RESCUER_KEYS:
                        step_now = RESCUER_KEYS[ev.key]
                    elif ev.key == pygame.K_g:  # L is LIDAR when you play the rescuer
                        viewer.layers["legend"] = not viewer.layers["legend"]
                    else:
                        handle_common_keys(viewer, ev.key)
                else:
                    if ev.key in ARROWS:
                        pending = ARROWS[ev.key]
                    elif ev.key == pygame.K_p:
                        paused = not paused
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        tps = min(tps * 1.5, 30)
                    elif ev.key == pygame.K_MINUS:
                        tps = max(tps / 1.5, 0.5)
                    else:
                        handle_common_keys(viewer, ev.key)
        if not obs.done:
            if side == "rescuer" and step_now is not None:
                obs = env.step(step_now)
                agent.observe(obs)
                agent.decision = {"reason": f"you chose {step_now}", "mode": "you"}
                receipts.append(receipt(obs.to_dict()))
                viewer.add_events(obs.t, events_of(obs.to_dict()))
                trail.append(list(env.pos))
            elif side == "target" and not paused:
                acc += dt * tps
                if acc >= 1:
                    acc = 0.0
                    tgt = None
                    if pending:
                        dx, dy = DIRS[pending]
                        tgt = (env.target.x + dx, env.target.y + dy)
                    pending = None
                    obs = env.step(action, target_move=tgt, target_controlled=True)
                    receipts.append(receipt(obs.to_dict()))
                    viewer.add_events(obs.t, events_of(obs.to_dict(), (agent.decision or {}).get("override")))
                    trail.append(list(env.pos))
                    action = agent.act(obs)   # a few ms; the event loop keeps running every frame
        fd = live_frame(env, agent, obs, trail, weather)
        viewer.draw(screen, fd, receipts, tick, hidden=(side == "rescuer" and not obs.done),
                    extra_help=help_r if side == "rescuer" else help_t)
        pygame.display.flip()
    pygame.quit()
