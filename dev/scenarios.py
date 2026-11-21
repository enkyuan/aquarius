"""Focused checks using the kit map and observations.

Run with python3 -m dev.scenarios --kit /path/to/mechfight-kit.
The ground loop approximates motion for a routing check; it does not score a match.
"""

import argparse
import gzip
import json
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from bot.arena import Arena, Prism
from bot.combat import FireControl, angles, contacts, intercept, muzzle
from bot.movement import Router, circle_at, distance, projectile_risk, steer
from bot.policy import Bot
from bot.tactics import Resources


def check_geometry():
    bridge = Prism(
        {
            "ring": [0, 0, 20, 0, 20, 20, 0, 20],
            "holes": [[7, 7, 13, 7, 13, 13, 7, 13]],
            "base": 10,
            "top": 12,
        }
    )
    assert bridge.intersects((-2, 10, 11), (3, 10, 11))
    assert not bridge.intersects((-2, 10, 8), (3, 10, 8))
    assert not bridge.intersects((8, 8, 11), (12, 12, 11))
    assert bridge.intersects((8, 8, 11), (16, 8, 11))
    assert abs(bridge.distance_sq(10, 10, 20) - 9) < 1e-9


def check_combat(obs, config, arena):
    t = intercept((0, 0, 0), (126, 0, 0), (0, 12, 0), 252, 3)
    assert t is not None and abs(math.hypot(126, t * 12) - (252 * t + 3)) < 1e-8
    obs = deepcopy(obs)
    me = obs["self"]
    center = muzzle(me)
    for angle in range(0, 360, 15):
        x = me["pos"][0] + 29 * math.cos(math.radians(angle))
        y = me["pos"][1] + 29 * math.sin(math.radians(angle))
        ground = arena.ground(x, y)
        if arena.body_clear(x, y, ground, 2.5, 9) and arena.shot_clear(
            center, (x, y, ground + 4.68)
        ):
            break
    else:
        raise AssertionError("no clear check position near the fixture spawn")
    me["aim"] = list(angles(center, (x, y, ground + 4.68)))
    target = {
        "id": 1,
        "kind": "mech",
        "pos": [x, y, ground],
        "vel": [0, 0, 0],
        "aim": [math.remainder(me["aim"][0] + math.pi, math.tau), 0],
        "hp": 1200,
        "max_hp": 1400,
        "weapons": ["gun"],
        "charging": False,
    }
    obs["mechs"] = [target]
    control = FireControl(config, arena)

    def choose():
        threats = contacts(obs, config, arena, {}, perf_counter() + 0.2)
        return control.decide(
            obs, threats, None, me["aim"], False, False, perf_counter() + 0.2
        )

    assert choose()["fire"] == [True, False]
    me["weapons"][1] = {
        "kind": "laser",
        "cooldown": 0,
        "charge": 0,
        "stats": config["weapons"]["laser"],
    }
    assert choose()["fire"] == [True, True]
    me["weapons"][1]["charge"] = config["weapons"]["laser"]["charge"]
    assert choose()["fire"] == [True, False]
    me["weapons"][1]["charge"] = 0
    obs["events"] = [{"type": "effect", "effect": "short", "by": 1}]
    control.update(obs)
    assert choose()["fire"] == [True, False]
    bullet = {"pos": [-20, 0, 4.5], "vel": [200, 0, 0]}
    assert projectile_risk((0, 0, 0), (0, 0, 0), 0, 0.12, [(bullet, 0)], 2.5, 9) > 0
    assert projectile_risk((0, 6, 0), (0, 7, 0), 0, 0.12, [(bullet, 0)], 2.5, 9) == 0


def check_resources(obs, config, arena):
    obs = deepcopy(obs)
    resources = Resources(config, arena)
    me = obs["self"]
    x, y, z = me["pos"]
    obs["pickups"] = [
        {"id": 8013, "pos": [x + 1, y, z + 9], "kind": "laser"},
        {"id": 8014, "pos": [x + 1, y, z], "kind": "lightning"},
    ]
    assert resources.take(obs, False) == {"take": 8014}
    me["weapons"][1] = {"kind": "lightning"}
    obs["pickups"][0]["pos"][2] = z
    assert resources.take(obs, False) == {"take": 8013, "drop": 1}
    me["hp"] = 880
    obs["drops"] = [
        {
            "id": 20000,
            "pos": [x, y, z],
            "holder": me["id"],
            "radius": 8,
            "lands_in": 0,
            "capture_s": 4,
            "progress": 2.2,
        }
    ]
    resources.update(obs)
    assert resources.objective(obs, [], False) == ((x, y), None)
    obs["offer"] = {
        "id": 1,
        "expires": 5,
        "choices": [{"id": "gun_damage"}, {"id": "nanites"}],
    }
    assert resources.upgrade(obs) == {"pick": 1}


def ground_loop(obs, arena):
    obs = deepcopy(obs)
    me = obs["self"]
    stats = me["stats"]
    goal = tuple(obs["zone"]["next_center"])
    router = Router(arena, stats["radius"], stats["height"])
    for tick in range(610):
        pos = me["pos"]
        if distance(pos, goal) < 4:
            return tick
        waypoint = router.target(pos, goal, stats, perf_counter() + 0.008)
        action = steer(
            arena, obs, waypoint, [], [], False, False, perf_counter() + 0.008
        )
        # Fixed-step ground approximation for checking routes around the recorded buildings.
        vx, vy, _ = me["vel"]
        dx, dy = [axis * stats["walk_speed"] for axis in action["move"]]
        ax, ay = dx - vx, dy - vy
        ratio = min(1, stats["ground_accel"] * 0.1 / max(0.001, math.hypot(ax, ay)))
        vx, vy = vx + ax * ratio, vy + ay * ratio
        for mx, my in ((vx, vy), (vx, 0), (0, vy)):
            x, y = pos[0] + mx * 0.1, pos[1] + my * 0.1
            z = arena.ground(x, y)
            if z - pos[2] <= stats["step_up"] and arena.body_clear(
                x, y, z, stats["radius"], stats["height"]
            ):
                me["pos"], me["vel"] = [x, y, z], [mx, my, 0]
                break
        else:
            me["vel"] = [0, 0, 0]
    raise AssertionError(
        f"ground loop stopped {distance(me['pos'], goal):.1f} m short at {me['pos']}"
    )


def check_dense(bot, obs):
    obs = deepcopy(obs)
    times = []
    for i in range(240):
        me = obs["self"]
        x, y, z = me["pos"]
        obs["mechs"] = [
            {
                "id": actor_id,
                "kind": "mech",
                "pos": [x + 3 * j - 22, y + 34, z],
                "vel": [-3, 4, 0],
                "aim": [-math.pi / 2, 0],
                "hp": 1000,
                "max_hp": 1400,
                "weapons": ["gun", "laser"],
                "charging": j == i % 15,
            }
            for j, actor_id in enumerate(
                actor_id for actor_id in range(16) if actor_id != me["id"]
            )
        ]
        obs["projectiles"] = [
            {
                "id": j + 1000,
                "kind": "mortar" if j % 6 == 0 else "gun",
                "pos": [x + 2 * j - 22, y + 29, z + 6],
                "vel": [0, -200, 0],
            }
            for j in range(25)
        ]
        obs["crates"] = [
            {"id": 1000 + j, "pos": [x + 2 * j, y - 22, z], "hp": 300}
            for j in range(36)
        ]
        obs["tick"], obs["t"] = i * 2, i / 10
        start = perf_counter()
        action = bot.act(obs)
        times.append(1000 * (perf_counter() - start))
        assert action["tick"] == obs["tick"] and len(action["fire"]) == 2
        assert all(math.isfinite(v) for v in action["aim"] + action["move"])
    return sorted(times)[237], max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.kit / "check" / "stream.jsonl.gz", "rt") as stream:
        hello, obs = json.loads(next(stream)), json.loads(next(stream))
    hello["map"]["dir"] = str(args.kit / "arena")
    config = hello["config"]
    arena = Arena(hello["map"]["dir"], config["mech"])
    check_geometry()
    check_combat(obs, config, arena)
    check_resources(obs, config, arena)
    assert circle_at(obs["zone"], 20, 70)[2] == 524
    ticks = ground_loop(obs, arena)
    p99, maximum = check_dense(Bot(hello), obs)
    print("PASS geometry, fire control, shot evasion, loot, supply drops, and upgrades")
    print(f"PASS ground routing loop in {ticks} decisions")
    print(
        f"PASS 240 synthetic dense decisions (host p99 {p99:.1f} ms, max {maximum:.1f} ms)"
    )


if __name__ == "__main__":
    main()
