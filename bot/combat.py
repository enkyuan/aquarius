"""Threat assessment, projectile leading, and weapon trigger management."""

import math
from dataclasses import dataclass
from time import perf_counter


@dataclass
class Contact:
    actor: dict
    meters: float
    pressure: float
    exposed: bool


def yaw_error(a, b):
    return abs(math.remainder(a - b, math.tau))


def angles(a, b):
    dx, dy, dz = (j - i for i, j in zip(a, b))
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def muzzle(me):
    x, y, z = me["pos"]
    return x, y, z + me["stats"]["muzzle_height"]


def cylinder(actor, config):
    kind = actor["kind"]
    if kind == "crate":
        return config["crates"]["radius"], config["crates"]["height"]
    if kind == "npc":
        return config["npc"]["radius"], config["npc"]["height"]
    stats = config["mech"]
    scale = 1 + config["cubes"].get("size_per_cube", 0.05) * actor.get("cubes", 0)
    return stats["radius"] * scale, stats["height"] * scale


def contacts(obs, config, arena, damage, deadline):
    me = obs["self"]
    result = []
    for actor in sorted(obs["mechs"], key=lambda e: math.dist(e["pos"], me["pos"]))[
        :10
    ]:
        if perf_counter() >= deadline:
            break
        meters = math.dist(me["pos"], actor["pos"])
        if meters > 293:
            continue
        x, y, z = actor["pos"]
        _, height = cylinder(actor, config)
        is_npc = actor["kind"] == "npc"
        if is_npc and meters > 72 and actor["id"] not in damage:
            continue
        start = (x, y, z + (2.6 if is_npc else height * 0.78))
        tx, ty, tz = me["pos"]
        end = tx, ty, tz + me["stats"]["height"] * 0.5
        exposed = arena.shot_clear(start, end)
        yaw, pitch = angles(start, end)
        aimed = (
            yaw_error(yaw, actor["aim"][0]) < 0.19
            and abs(pitch - actor["aim"][1]) < 0.24
        )
        dx, dy = tx - x, ty - y
        closing = (dx * actor["vel"][0] + dy * actor["vel"][1]) / max(meters, 1)
        pressure = (0.5 if is_npc else 1.2) * max(0.0, 1.6 - meters / 150)
        if exposed:
            pressure += (0.65 if is_npc else 1.2) if aimed else 0.2
            if actor.get("charging"):
                pressure += 1.9
            if meters < 53 and "lightning" in actor.get("weapons", ()):
                pressure += 1.8
        if actor["id"] in damage:
            pressure += 1.3 if exposed else 0.45
        if closing > 4 and not is_npc:
            pressure += 0.35
        result.append(Contact(actor, meters, pressure, exposed))
    return sorted(result, key=lambda c: c.pressure, reverse=True)


def intercept(origin, point, velocity, speed, offset):
    """Time to the first intersection with the expanding muzzle-offset sphere."""
    rx, ry, rz = (b - a for a, b in zip(origin, point))
    vx, vy, vz = velocity
    a = vx * vx + vy * vy + vz * vz - speed * speed
    b = 2 * (rx * vx + ry * vy + rz * vz - speed * offset)
    c = rx * rx + ry * ry + rz * rz - offset * offset
    if c <= 0:
        return 0.0
    if abs(a) < 1e-9:
        return -c / b if b < 0 else None
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    times = ((-b - root) / (2 * a), (-b + root) / (2 * a))
    return min((t for t in times if t >= 0), default=None)


@dataclass
class Shot:
    aim: tuple
    point: tuple
    meters: float
    radius: float


def shot_at(arena, me, actor, weapon, config, dt):
    origin = muzzle(me)
    radius, height = cylinder(actor, config)
    kind = weapon["kind"]
    arc = kind == "lightning"
    speed = weapon["stats"].get("speed") if kind == "gun" else None
    reach = weapon["stats"]["range"]
    offset = me["stats"]["radius"] + 0.5
    delay = max(dt, weapon["cooldown"]) if speed else dt
    # Far-future lead is useful only when the next shot can be evaluated soon.
    delay = min(delay, dt * 2)
    velocity = actor["vel"]
    vx, vy, vz = velocity
    x, y, z = (p + v * delay for p, v in zip(actor["pos"], velocity))
    origin = tuple(p + v * delay for p, v in zip(origin, me["vel"]))
    if arc:
        heights = (0.5,)
    else:
        heights = (0.52, 0.72, 0.34)
    for fraction in heights:
        point = x, y, z + height * fraction
        if speed:
            t = intercept(origin, point, velocity, speed, offset)
            if t is None or speed * t > reach:
                continue
            point = point[0] + vx * t, point[1] + vy * t, point[2] + vz * t
        meters = math.dist(origin, point)
        if meters > reach + (offset if speed else 0) or not arena.shot_clear(
            origin, point
        ):
            continue
        aim = angles(origin, point)
        stats = me["stats"]
        if arc:
            aim = aim[0], min(stats["max_pitch"], max(stats["min_pitch"], aim[1]))
        elif not stats["min_pitch"] <= aim[1] <= stats["max_pitch"]:
            continue
        return Shot(aim, point, meters, radius)
    return None


def turns_in(me, aim):
    stats = me["stats"]
    return max(
        yaw_error(aim[0], me["aim"][0]) / stats["turn_rate"],
        abs(aim[1] - me["aim"][1]) / stats["pitch_rate"],
    )


def aligned(me, current, desired, shot, weapon, time):
    stats = me["stats"]
    yaw_step = stats["turn_rate"] * time
    pitch_step = stats["pitch_rate"] * time
    yd = math.remainder(desired[0] - current[0], math.tau)
    pd = desired[1] - current[1]
    yaw = current[0] + min(yaw_step, max(-yaw_step, yd))
    pitch = current[1] + min(pitch_step, max(-pitch_step, pd))
    if weapon["kind"] == "lightning":
        return (
            yaw_error(yaw, shot.aim[0])
            <= math.radians(weapon["stats"]["cone_deg"]) * 0.52
        )
    tolerance = math.atan2(shot.radius * 0.7, max(shot.meters, 1))
    return (
        yaw_error(yaw, shot.aim[0]) <= tolerance
        and abs(pitch - shot.aim[1]) <= tolerance
    )


def arc_blocked(arena, obs, target, shot, weapon):
    origin = muzzle(obs["self"])
    target_distance = math.dist(origin, shot.point)
    reach = min(target_distance, weapon["stats"]["range"])
    for actor in (*obs["mechs"], *obs["crates"]):
        if actor["id"] == target["id"] or math.dist(actor["pos"], origin) >= reach:
            continue
        x, y, z = actor["pos"]
        height = (
            2
            if "hp" in actor and "kind" not in actor
            else (1.8 if actor.get("kind") == "npc" else 4.5)
        )
        center = x, y, z + height
        if (
            yaw_error(angles(origin, center)[0], shot.aim[0])
            < math.radians(weapon["stats"]["cone_deg"])
        ) and arena.shot_clear(origin, center):
            return True
    return False


class FireControl:
    def __init__(self, config, arena):
        self.config = config
        self.arena = arena
        self.dt = 1 / config["match"]["tick_hz"]
        self.horizon = self.dt * config["match"]["decide_every"]
        self.short_until = -math.inf

    def update(self, obs):
        for event in obs["events"]:
            if event.get("effect") == "short" and event["type"] == "effect":
                self.short_until = (
                    obs["t"] + self.config["weapons"]["lightning"]["short_s"]
                )

    def decide(self, obs, threats, farm, fallback, retreat, quiet, deadline):
        me = obs["self"]
        action = {"aim": list(fallback), "fire": [False, False]}
        slots = [
            (i, w) for i, w in enumerate(me["weapons"]) if w and w["kind"] != "mortar"
        ]
        charging = any(
            w["kind"] == "laser" and w.get("charge", 0) > 0 for _, w in slots
        )
        gun = next((w for _, w in slots if w["kind"] == "gun"), None)
        if (
            gun
            and gun.get("reload", 0) <= 0
            and gun.get("ammo", 0) < gun["stats"]["mag"]
        ):
            action["reload"] = (quiet or retreat or gun["ammo"] == 0) and (
                gun["ammo"] <= 6 or not threats
            )
        if quiet or perf_counter() >= deadline:
            return action
        candidates = [
            c.actor
            for c in threats
            if c.exposed or not retreat
            if c.actor["kind"] == "mech" or (c.exposed and c.meters < 43)
        ][:5]
        if (
            farm
            and not retreat
            and all(actor["id"] != farm["id"] for actor in candidates)
        ):
            candidates.append(farm)
        options = []
        for actor in candidates:
            if perf_counter() >= deadline:
                break
            for slot, weapon in slots:
                kind = weapon["kind"]
                if kind == "laser" and (
                    obs["t"] < self.short_until or (retreat and not charging)
                ):
                    continue
                if kind == "lightning" and me["energy"] < weapon["stats"]["energy"] + (
                    8 if retreat else 16
                ):
                    continue
                shot = shot_at(self.arena, me, actor, weapon, self.config, self.dt)
                if not shot or (
                    kind == "lightning"
                    and arc_blocked(self.arena, obs, actor, shot, weapon)
                ):
                    continue
                turn = turns_in(me, shot.aim)
                finish = actor.get("hp", 1500)
                value = 23 if actor["kind"] == "mech" else 2
                value += (
                    13 * (1 - min(1.0, finish / actor.get("max_hp", 1400)))
                    if actor["kind"] == "mech"
                    else 0
                )
                value -= 0.045 * shot.meters + 9 * turn + 5 * min(2, weapon["cooldown"])
                damage = weapon["stats"]["damage"]
                if kind == "laser":
                    value += 6 if not retreat else 0
                    if "mortar" in actor.get("weapons", ()):
                        damage *= 1 + weapon["stats"]["cookoff"]
                        value += 5
                    if weapon.get("charge", 0):
                        value += 11
                elif kind == "lightning":
                    value += 8 if "laser" in actor.get("weapons", ()) else 4
                elif (
                    weapon.get("ammo", 0) <= 0 or weapon.get("reload", 0) > self.horizon
                ):
                    value -= 12
                if damage >= finish:
                    value += 10
                options.append((value, slot, weapon, actor, shot))
        if not options:
            return action
        _, _, _, target, main = max(options, key=lambda item: item[0])
        action["aim"] = list(main.aim)
        for _, slot, weapon, actor, shot in options:
            if actor["id"] != target["id"]:
                continue
            kind = weapon["kind"]
            next_fire = min(self.horizon, max(self.dt, weapon["cooldown"]))
            ready = weapon["cooldown"] <= self.horizon + 1e-5
            if kind == "laser":
                charge, stats = weapon.get("charge", 0), weapon["stats"]
                if charge:
                    fraction = stats["min_frac"] + (1 - stats["min_frac"]) * min(
                        1.0, charge / stats["charge"]
                    )
                    damage = stats["damage"] * fraction
                    if "mortar" in actor.get("weapons", ()):
                        damage *= 1 + stats["cookoff"]
                    accurate = aligned(me, me["aim"], main.aim, shot, weapon, self.dt)
                    finish = actor.get("hp", math.inf) <= damage
                    full = charge + 1e-5 >= stats["charge"]
                    fire = not (
                        retreat
                        or (
                            accurate
                            and (full or (charge >= stats["min_charge"] and finish))
                        )
                    )
                    if turns_in(me, shot.aim) > 0.35 and full:
                        fire = False
                else:
                    # In a close arc duel, preserve full movement and use the gun.
                    safe_to_charge = not retreat and not (
                        shot.meters < 52 and "lightning" in actor.get("weapons", ())
                    )
                    fire = ready and safe_to_charge and turns_in(me, main.aim) < 0.32
                action["fire"][slot] = fire
            else:
                loaded = kind != "gun" or (
                    weapon.get("ammo", 0) > 0 and weapon.get("reload", 0) <= next_fire
                )
                if (
                    ready
                    and loaded
                    and aligned(me, me["aim"], main.aim, shot, weapon, next_fire)
                ):
                    action["fire"][slot] = True
                    if kind == "gun":
                        action["reload"] = False
        return action
