"""Closed-loop first-weapon regression on a recorded map lane (no opponents)."""

import argparse
import gzip
import json
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from bot.movement import distance
from bot.policy import Bot


def run_lane(bot, observation, item, warning=False):
    obs = deepcopy(observation)
    me, config = obs["self"], bot.config
    stats = me["stats"]
    dt = 1 / config["match"]["tick_hz"]
    ticks = config["match"]["decide_every"]
    start_tick = obs["tick"]
    # Isolate one known, visible pickup. This experiment models ground motion,
    # energy and acquisition. It does not simulate opponents or perception.
    obs.update(
        pickups=[item],
        crates=[],
        cubes=[],
        drops=[],
        mechs=[],
        projectiles=[],
        events=[],
    )
    min_energy = me["energy"]
    last_spend = -math.inf
    boosts = 0
    for decision in range(240):
        obs["tick"] = start_tick + ticks * decision
        obs["t"] = obs["tick"] * dt
        obs["sounds"] = (
            [{"kind": "gun", "dist": "near", "bearing": 0, "age": 0}] if warning else []
        )
        action = bot.act(obs)
        if action.get("take") == item["id"]:
            assert distance(me["pos"], item["pos"]) <= config["loot"]["pickup_radius"]
            assert abs(me["pos"][2] - item["pos"][2]) <= 4
            return decision * ticks * dt, me["energy"], min_energy, boosts
        boosts += action["boost"]
        speed = stats["boost_speed"] if action["boost"] else stats["walk_speed"]
        assert not action["jump"], "fixture left the ground model"
        for k in range(ticks):
            x, y, z = me["pos"]
            vx, vy, _ = me["vel"]
            ax, ay = (v * speed - u for v, u in zip(action["move"], (vx, vy)))
            f = min(1, stats["ground_accel"] * dt / max(1e-9, math.hypot(ax, ay)))
            vx, vy = vx + ax * f, vy + ay * f
            nx, ny = x + vx * dt, y + vy * dt
            assert bot.arena.walk_clear((x, y, z), (nx, ny), stats)
            me["pos"] = [nx, ny, bot.arena.ground(nx, ny)]
            me["vel"] = [vx, vy, 0]
            t = obs["t"] + dt * k
            if action["boost"]:
                me["energy"] -= stats["boost_drain"] * dt
                last_spend = t + dt
            elif t - last_spend + 1e-9 >= stats["energy_regen_delay"]:
                me["energy"] = min(
                    me["max_energy"], me["energy"] + stats["energy_regen"] * dt
                )
            min_energy = min(min_energy, me["energy"])
            assert min_energy >= 0
    raise AssertionError("bot did not take the pickup")


def sweep(hello, messages, obs):
    """Six fixed 75 m approaches to each recorded ground weapon."""
    bot = Bot(hello)
    arena = bot.arena
    items = {
        p["id"]: p
        for frame in messages
        for p in frame.get("pickups", [])
        if p["kind"] in ("laser", "lightning")
    }
    gains = []
    for item in items.values():
        gx, gy, gz = item["pos"]
        if abs(gz - arena.ground(gx, gy)) > 4:
            continue
        for degrees in range(0, 360, 60):
            angle = math.radians(degrees)
            x, y = gx + 75 * math.cos(angle), gy + 75 * math.sin(angle)
            z = arena.ground(x, y)
            frame = deepcopy(obs)
            me = frame["self"]
            me.update(
                pos=[x, y, z],
                vel=[0, 0, 0],
                aim=[math.remainder(angle + math.pi, math.tau), 0],
            )
            stats = me["stats"]
            if not arena.body_clear(
                x, y, z, stats["radius"], stats["height"]
            ) or not bot.opening.lane(me["pos"], item["pos"], stats, math.inf):
                continue
            baseline = Bot(hello)
            with patch.object(baseline.opening, "sprint", return_value=None):
                slow = run_lane(baseline, frame, item)
            fast = run_lane(Bot(hello), frame, item)
            assert fast[1] == slow[1] == me["max_energy"]
            gains.append(round(slow[0] - fast[0], 1))
    print(
        f"{len(gains)} clear approaches: {sum(g > 0 for g in gains)} faster, "
        f"{gains.count(0)} ties, {sum(g < 0 for g in gains)} slower, "
        f"mean pickup time saved {sum(gains) / len(gains):.2f}s"
    )


def check(kit, run_sweep):
    with gzip.open(kit / "check" / "stream.jsonl.gz", "rt") as stream:
        hello, *messages = map(json.loads, stream)
    hello["map"]["dir"] = str(kit / "arena")
    obs = next(o for o in messages if o.get("t") == 3.8)
    item = next(p for p in obs["pickups"] if p["id"] == 8022)
    baseline = Bot(hello)
    with patch.object(baseline.opening, "sprint", return_value=None):
        slow = run_lane(baseline, obs, item)
    fast = run_lane(Bot(hello), obs, item)
    interrupted = run_lane(Bot(hello), obs, item, warning=True)
    assert fast[0] < slow[0] and fast[1] == slow[1] == obs["self"]["max_energy"]
    assert fast[3] > 0 and interrupted[3] == 0 and slow[3] == 0
    for label, case in (
        ("walk", slow),
        ("opening", fast),
        ("gun warning", interrupted),
    ):
        print(
            f"{label:12} take {case[0]:.1f}s   arrival energy {case[1]:.1f}   minimum {case[2]:.1f}   boost decisions {case[3]}"
        )

    if run_sweep:
        sweep(hello, messages, obs)

    bot = Bot(hello)

    def sprint(frame):
        return bot.opening.sprint(frame, item, perf_counter() + 1)

    assert sprint(obs)
    with patch.object(bot.opening, "wall_cells", {}):
        assert sprint(obs) is None
    with patch.object(bot.opening, "wall_at", return_value=False):
        assert sprint(obs) is None
    poor = deepcopy(obs)
    poor["self"]["energy"] = 10
    assert sprint(poor) is None
    armed = deepcopy(obs)
    armed["self"]["weapons"][1] = {"kind": "laser"}
    assert sprint(armed) is None
    closing = deepcopy(obs)
    closing["zone"]["shrink_end"] = obs["t"] + 1
    assert sprint(closing) is None
    invisible = deepcopy(obs)
    invisible["pickups"] = []
    invisible["tick"] += hello["config"]["match"]["decide_every"]
    invisible["t"] = invisible["tick"] / hello["config"]["match"]["tick_hz"]
    bot.act(obs)
    assert not bot.act(invisible)["boost"]
    alert = next(o for o in messages if o.get("t") == 3.9)
    bot.opening.observe(alert)
    assert sprint(alert) is None
    print(
        "PASS wall access, energy reserve, zone deadline, occupied slot, lost pickup and recorded interruption"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Test independent approach angles on the kit map",
    )
    args = parser.parse_args()
    check(args.kit, args.sweep)
