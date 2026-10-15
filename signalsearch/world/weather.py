"""Weather: fetched once from Open-Meteo (live) or taken from the config (fixed).

The fetched numbers are written into the resolved config, so a run replays offline.
"""
from __future__ import annotations

import copy
import time

import requests

API = "https://api.open-meteo.com/v1/forecast"


def fetch_live(lat: float, lon: float, timeout: float) -> dict:
    r = requests.get(API, params={"latitude": lat, "longitude": lon,
                                  "current": "temperature_2m,relative_humidity_2m,rain"}, timeout=timeout)
    r.raise_for_status()
    cur = r.json()["current"]
    return {"temperature": float(cur["temperature_2m"]), "humidity": float(cur["relative_humidity_2m"]),
            "rain": float(cur.get("rain") or 0.0), "time": cur.get("time")}


def resolve_weather(cfg: dict, log=print) -> dict:
    """Return cfg with world.weather.resolved filled in (no-op if already resolved)."""
    cfg = copy.deepcopy(cfg)
    w = cfg["world"]["weather"]
    if "resolved" in w:
        return cfg
    loc = w["location"]
    fixed = dict(w["fixed"][loc])
    if w["source"] == "live":
        lat, lon = w["coords"][loc]
        res, err = None, None
        for attempt in range(int(w.get("retries", 3))):  # one slow response must not cost the live weather
            try:
                res = fetch_live(lat, lon, w.get("timeout_s", 10))
                res["origin"] = f"open-meteo live ({loc})"
                break
            except Exception as e:  # network down, API change, ...
                err = e
                time.sleep(1.5 * (attempt + 1))
        if res is None:
            log(f"weather: live fetch failed ({err!r}); using fixed {loc} values")
            res = {**fixed, "origin": f"fixed fallback ({loc}) after fetch failure"}
    else:
        res = {**fixed, "origin": f"fixed ({loc})"}
    res["location"] = loc
    w["resolved"] = res
    return cfg
