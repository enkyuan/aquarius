"""Paired local gun/laser scrimmages using the kit map and recorded encounter positions.

The scripted opponent and simplified motion are independent of the practice scorer.
Use the output to investigate fight behavior. It has no placement-point scale.
"""

import argparse
import gzip
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from bot.policy import Bot


def wrap(angle):
    return math.remainder(angle, math.tau)


def limit(value, span):
    return max(-span, min(span, value))


def direction(aim):
    yaw, pitch = aim
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )


def point(start, vector, distance):
    return tuple(a + b * distance for a, b in zip(start, vector))


def angles(start, end):
    x, y, z = (b - a for a, b in zip(start, end))
    return math.atan2(y, x), math.atan2(z, math.hypot(x, y))


def cylinder_impact(start, end, foe, padding=0):
    """Clip the segment to the horizontal circle and vertical height intervals."""
    x, y, z = (a - b for a, b in zip(start, foe["pos"]))
    dx, dy, dz = (b - a for a, b in zip(start, end))
    radius, height = foe["stats"]["radius"] + padding, foe["stats"]["height"]
    a, b, c = dx * dx + dy * dy, x * dx + y * dy, x * x + y * y - radius * radius
    low, high = 0.0, 1.0
    if a:
        disc = b * b - a * c
        if disc < 0:
            return None
        root = math.sqrt(disc)
        low, high = max(low, (-b - root) / a), min(high, (-b + root) / a)
    elif c > 0:
        return None
    if dz:
        t1, t2 = (-padding - z) / dz, (height + padding - z) / dz
        low, high = max(low, min(t1, t2)), min(high, max(t1, t2))
    elif not -padding <= z <= height + padding:
        return None
    return point(start, (dx, dy, dz), low) if low <= high else None


def body(pose, config, actor_id, laser, hp):
    weapons = config["weapons"]
    gun = {
        "kind": "gun",
        "cooldown": 0.0,
        "ammo": weapons["gun"]["mag"],
        "reload": 0.0,
        "stats": weapons["gun"],
    }
    secondary = (
        {"kind": "laser", "cooldown": 0.0, "charge": 0.0, "stats": weapons["laser"]}
        if laser
        else None
    )
    stats = config["mech"]
    return {
        "id": actor_id,
        "pos": list(pose["pos"]),
        "vel": [0.0, 0.0, 0.0],
        "aim": list(pose["aim"]),
        "hp": hp,
        "max_hp": stats["hp"],
        "grounded": True,
        "energy": stats["energy"],
        "max_energy": stats["energy"],
        "cubes": 0,
        "upgrades": [],
        "in_zone": True,
        "stats": stats,
        "weapons": [gun, secondary],
    }


@dataclass
class Projectile:
    owner: int
    pos: tuple
    vel: tuple
    damage: float
    remaining: float
    serial: int


class Duel:
    def __init__(self, bot, hello, fixture, own_laser, enemy_laser, seed):
        self.bot, self.arena, self.config = bot, bot.arena, hello["config"]
        self.dt = 1 / self.config["match"]["tick_hz"]
        self.interval = int(self.config["match"]["decide_every"])
        pose = fixture["self"]
        foe = next(m for m in fixture["mechs"] if m["kind"] == "mech")
        self.players = [
            body(pose, self.config, pose["id"], own_laser, pose["hp"]),
            body(
                foe, self.config, foe["id"], enemy_laser, foe.get("hp", pose["max_hp"])
            ),
        ]
        for me in self.players:
            me["pos"][2] = self.arena.ground(*me["pos"][:2])
        self.tick = 0
        self.now = 0.0
        self.serial = 0
        self.projectiles = []
        self.events = []
        self.shots = [0, 0]
        self.hits = [0, 0]
        self.combat = [-math.inf, -math.inf]
        self.energy = [-math.inf, -math.inf]
        self.rng = [random.Random(seed + offset) for offset in (145, 291, 410)]
        self.aim_error = self.rng[2].uniform(0.002, 0.016)
        self.strafe_period = self.rng[2].uniform(0.8, 2.0)
        self.aggression = self.rng[2].uniform(0.58, 1.0)
        self.side = self.rng[2].choice((-1, 1))
        self.zone = dict(fixture["zone"], shrink_start=1000, shrink_end=1100)

    def observation(self):
        me, foe = self.players
        enemy = {
            key: foe[key]
            for key in ("id", "pos", "vel", "aim", "hp", "max_hp", "cubes")
        }
        enemy.update(
            kind="mech",
            charging=bool(foe["weapons"][1] and foe["weapons"][1]["charge"]),
            weapons=[w["kind"] for w in foe["weapons"] if w],
        )
        return {
            "type": "obs",
            "tick": self.tick,
            "t": self.now,
            "self": me,
            "alive": 2,
            "mechs": [enemy],
            "crates": [],
            "cubes": [],
            "pickups": [],
            "drops": [],
            "sounds": [],
            "beams": [],
            "offer": None,
            "events": self.events,
            "zone": self.zone,
            "projectiles": [
                {"id": p.serial, "kind": "gun", "pos": p.pos, "vel": p.vel}
                for p in self.projectiles
                if p.owner == 1
            ],
        }

    def opponent(self):
        foe, me = self.players
        stats = me["stats"]
        origin = me["pos"][0], me["pos"][1], me["pos"][2] + stats["muzzle_height"]
        toward = angles(origin, foe["pos"])[0]
        meters = math.dist(me["pos"], foe["pos"])
        laser = me["weapons"][1]
        firing_laser = bool(
            laser and laser["charge"] >= laser["stats"]["charge"] - self.dt / 2
        )
        t = 0 if firing_laser else meters / me["weapons"][0]["stats"]["speed"]
        target = (
            foe["pos"][0],
            foe["pos"][1],
            foe["pos"][2] + foe["stats"]["height"] * 0.52,
        )
        aim = angles(origin, point(target, foe["vel"], t))
        aim = [a + self.rng[2].gauss(0, self.aim_error) for a in aim]
        forward = limit((meters - 18) / 34, self.aggression)
        side = (
            self.side
            * (-1) ** int(self.now / self.strafe_period)
            * math.sqrt(1 - forward**2)
            * 0.68
        )
        move = [
            forward * math.cos(toward) - side * math.sin(toward),
            forward * math.sin(toward) + side * math.cos(toward),
        ]
        clear = self.arena.shot_clear(origin, target)
        hold_laser = bool(laser and (clear or laser["charge"]) and not firing_laser)
        boost = not clear and me["energy"] > 30
        return {
            "aim": aim,
            "move": move,
            "fire": [clear and not firing_laser, hold_laser],
            "boost": boost,
        }

    def move(self, index, action):
        me = self.players[index]
        stats = me["stats"]
        dt = self.dt
        x, y, z = me["pos"]
        vx, vy, vz = me["vel"]
        grounded = me["grounded"]
        ground_boost = bool(action.get("boost") and grounded and me["energy"] > 1)
        if ground_boost:
            me["energy"] -= dt * stats["boost_drain"]
            self.energy[index] = self.now
        elif self.now - self.energy[index] >= stats["energy_regen_delay"]:
            me["energy"] = min(
                me["max_energy"], me["energy"] + dt * stats["energy_regen"]
            )
        if grounded and action.get("jump"):
            grounded, vz = False, stats["jump_speed"]
        speed = stats["boost_speed"] if ground_boost else stats["walk_speed"]
        if me["weapons"][1] and me["weapons"][1]["charge"]:
            speed *= stats["charge_move_mult"]
        mx, my = action.get("move", (0, 0))
        speed /= max(1.0, math.hypot(mx, my))
        ax, ay = mx * speed - vx, my * speed - vy
        accel = stats["ground_accel"] if grounded else stats["air_accel"]
        blend = min(1, dt * accel / max(0.00001, math.hypot(ax, ay)))
        vx, vy = vx + ax * blend, vy + ay * blend
        if not grounded:
            vz -= stats["gravity"] * dt
        for ux, uy in ((vx, vy), (vx, 0), (0, vy), (0, 0)):
            nx, ny = x + dt * ux, y + dt * uy
            ground = self.arena.ground(nx, ny)
            effective_z = ground if grounded else max(ground, z + dt * vz)
            if effective_z - z > stats["step_up"] and grounded:
                continue
            if not self.arena.body_clear(
                nx, ny, effective_z, stats["radius"], stats["height"]
            ):
                continue
            grounded = effective_z == ground
            me["pos"] = [nx, ny, effective_z]
            me["vel"] = [ux, uy, 0 if grounded else vz]
            me["grounded"] = grounded
            break
        yaw, pitch = me["aim"]
        target_yaw, target_pitch = action.get("aim", me["aim"])
        target_pitch = min(stats["max_pitch"], max(stats["min_pitch"], target_pitch))
        me["aim"] = [
            wrap(yaw + limit(wrap(target_yaw - yaw), dt * stats["turn_rate"])),
            pitch + limit(target_pitch - pitch, dt * stats["pitch_rate"]),
        ]

    def damage(self, shooter, damage, kind, source):
        target = self.players[1 - shooter]
        bearing = angles(target["pos"], source)[0]
        if abs(wrap(bearing - target["aim"][0])) > math.radians(
            target["stats"]["front_arc_deg"] / 2
        ):
            damage *= target["stats"]["rear_mult"]
        target["hp"] = max(0.0, target["hp"] - damage)
        self.combat[:] = [self.now, self.now]
        self.hits[shooter] += 1
        if shooter == 0:
            self.events.append(
                {
                    "type": "hit",
                    "target": target["id"],
                    "weapon": kind,
                    "amount": damage,
                }
            )
        else:
            self.events.append(
                {
                    "type": "damaged",
                    "by": self.players[shooter]["id"],
                    "bearing": bearing,
                    "weapon": kind,
                    "amount": damage,
                }
            )

    def shoot(self, index, action):
        me = self.players[index]
        gun, laser = me["weapons"]
        dt = self.dt
        fire = action.get("fire", (False, False))
        gun["cooldown"] = max(0.0, gun["cooldown"] - dt)
        reload = gun["reload"]
        if reload > 0:
            gun["reload"] = max(0.0, reload - dt)
            if gun["reload"] < 1e-8:
                gun["ammo"] = gun["stats"]["mag"]
                gun["reload"] = 0.0
        elif action.get("reload") and gun["ammo"] < gun["stats"]["mag"]:
            gun["reload"] = gun["stats"]["reload"]
        elif fire[0] and gun["ammo"] and gun["cooldown"] < 1e-8:
            stats = gun["stats"]
            spread = math.radians(stats["spread_deg"])
            aim = [a + self.rng[index].gauss(0, spread) for a in me["aim"]]
            vector = direction(aim)
            x, y, z = me["pos"]
            muzzle = x, y, z + me["stats"]["muzzle_height"]
            start = point(muzzle, vector, me["stats"]["radius"] + 0.5)
            self.shots[index] += 1
            gun["ammo"] -= 1
            gun["cooldown"] = stats["cooldown"]
            if gun["ammo"] == 0:
                gun["reload"] = stats["reload"]
            if self.arena.shot_clear(muzzle, start):
                self.serial += 1
                vel = tuple(a * stats["speed"] for a in vector)
                self.projectiles.append(
                    Projectile(
                        index, start, vel, stats["damage"], stats["range"], self.serial
                    )
                )
        if not laser:
            return
        laser["cooldown"] = max(0.0, laser["cooldown"] - dt)
        if laser["cooldown"] > 1e-8:
            return
        stats = laser["stats"]
        if fire[1]:
            laser["charge"] += dt
            if laser["charge"] < stats["charge"] + stats["hold_s"] - 1e-8:
                return
        elif not laser["charge"]:
            return
        if laser["charge"] + 1e-8 < stats["min_charge"]:
            laser["charge"] += dt
            return
        frac = min(1, laser["charge"] / stats["charge"])
        damage = stats["damage"] * (stats["min_frac"] + (1 - stats["min_frac"]) * frac)
        laser["charge"], laser["cooldown"] = 0, stats["cooldown"]
        x, y, z = me["pos"]
        muzzle = x, y, z + me["stats"]["muzzle_height"]
        vector = direction(me["aim"])
        self.shots[index] += 1
        impact = cylinder_impact(
            muzzle, point(muzzle, vector, stats["range"]), self.players[1 - index]
        )
        if impact and self.arena.shot_clear(muzzle, impact):
            self.damage(index, damage, "laser", muzzle)

    def step(self, actions):
        for index, action in enumerate(actions):
            self.move(index, action)
        for index, action in enumerate(actions):
            if self.players[index]["hp"] > 0:
                self.shoot(index, action)
        active = []
        for p in self.projectiles:
            speed = math.hypot(*p.vel)
            dt = min(self.dt, p.remaining / speed)
            pos = point(p.pos, p.vel, dt)
            impact = cylinder_impact(
                p.pos,
                pos,
                self.players[1 - p.owner],
                self.config["projectile"]["gun_radius"],
            )
            if impact is not None:
                if self.arena.shot_clear(p.pos, impact):
                    self.damage(p.owner, p.damage, "gun", p.pos)
            elif p.remaining > self.dt * speed and self.arena.shot_clear(p.pos, pos):
                p.pos = pos
                p.remaining -= self.dt * speed
                active.append(p)
        self.projectiles = active
        for i, me in enumerate(self.players):
            if me["hp"] > 0 and self.now - self.combat[i] >= me["stats"]["regen_delay"]:
                me["hp"] = min(
                    me["max_hp"], me["hp"] + self.dt * me["stats"]["regen_per_s"]
                )
        self.tick += 1
        self.now = self.tick * self.dt

    def run(self, seconds):
        while self.now < seconds and min(m["hp"] for m in self.players) > 0:
            if self.tick % self.interval == 0:
                actions = (self.bot.act(self.observation()), self.opponent())
                self.events = []
            self.step(actions)
        own, enemy = (m["hp"] for m in self.players)
        return {
            "hp": own,
            "opponent_hp": enemy,
            "seconds": self.now,
            "shots": self.shots[0],
            "hits": self.hits[0],
            "result": (own > enemy) - (own < enemy),
        }


def run_batch(kit, seed_count, seconds, seed_offset):
    with gzip.open(kit / "check" / "stream.jsonl.gz", "rt") as stream:
        hello, *stream = map(json.loads, stream)
    hello["map"]["dir"] = str((kit / "arena").resolve())
    recorded = {o["tick"]: o for o in stream if o["type"] == "obs"}
    scenes = {
        "gun_vs_laser": (142, False, True),
        "low_hp_corner": (376, False, True),
        "laser_vs_gun": (142, True, False),
        "laser_duel": (718, True, True),
    }
    out = {}
    for name, (tick, own_laser, enemy_laser) in scenes.items():
        fixture = recorded[tick]
        cases = []
        for seed in range(seed_offset, seed_offset + seed_count):
            config = {**hello, "seed": seed}
            bot = Bot(config)
            cases.append(
                Duel(bot, config, fixture, own_laser, enemy_laser, seed).run(seconds)
            )
        out[name] = cases
    return out


def archive(ref, path):
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=root,
        text=True,
    ).strip()
    tar_path = path / "tree.tar"
    with tar_path.open("wb") as dest:
        subprocess.run(["git", "archive", commit], cwd=root, check=True, stdout=dest)
    with tarfile.open(tar_path) as source:
        source.extractall(path, filter="data")


def show(label, data):
    print(label)
    for name, cases in data.items():
        wins = sum(c["result"] == 1 for c in cases)
        ties = sum(c["result"] == 0 for c in cases)
        n = len(cases)
        hp, foe_hp, hits, shots, seconds = (
            fmean(case[key] for case in cases)
            for key in ("hp", "opponent_hp", "hits", "shots", "seconds")
        )
        print(
            f"  {name:15} {wins:2}/{n} wins, {ties:2} ties, "
            f"mean HP {hp:6.1f}/{foe_hp:6.1f}, "
            f"hits/shots {hits:4.1f}/{shots:4.1f}, duration {seconds:4.1f} s"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--against", help="Compare the working tree to a committed Git revision."
    )
    parser.add_argument("--raw", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seeds < 1 or args.seconds <= 0 or not math.isfinite(args.seconds):
        parser.error("seeds and seconds must be positive and finite")
    if args.raw:
        print(
            json.dumps(run_batch(args.kit, args.seeds, args.seconds, args.seed_offset))
        )
        return
    if args.against:
        with tempfile.TemporaryDirectory(prefix="aquarius-duels-") as folder:
            path = Path(folder)
            archive(args.against, path)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--kit",
                str(args.kit.resolve()),
                "--seeds",
                str(args.seeds),
                "--seconds",
                str(args.seconds),
                "--seed-offset",
                str(args.seed_offset),
                "--raw",
            ]
            env = {**os.environ, "PYTHONPATH": str(path)}
            prior = json.loads(subprocess.check_output(command, env=env, cwd=path))
        show(args.against, prior)
    current = run_batch(args.kit, args.seeds, args.seconds, args.seed_offset)
    show("working tree", current)
    if args.against:
        margins = [
            (new["hp"] - new["opponent_hp"]) - (old["hp"] - old["opponent_hp"])
            for name, cases in current.items()
            for old, new in zip(prior[name], cases)
        ]
        print(
            f"Paired HP-margin change: {sum(margins) / len(margins):+.1f} HP over {len(margins)} scenarios"
        )
    print("Local duel outcomes only. Practice placement requires the official scorer.")


if __name__ == "__main__":
    main()
