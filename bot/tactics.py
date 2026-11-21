"""Score reachable resources and translate loadout and upgrade choices."""

import math

from .movement import distance, zone_margin


def weapon_priority(kind, close_laser=False):
    if kind == "laser":
        return 8 if close_laser else 12
    if kind == "lightning":
        return 12 if close_laser else 8
    return 0


def pickup_slot(me, kind, close_laser=False):
    weapons = me["weapons"]
    if not weapon_priority(kind, close_laser) or any(
        w and w["kind"] == kind for w in weapons
    ):
        return None
    for i, weapon in enumerate(weapons):
        if weapon is None:
            return i
    for i, weapon in enumerate(weapons):
        if (
            weapon["kind"] != "gun"
            and not weapon.get("charge", 0)
            and weapon_priority(kind, close_laser)
            > weapon_priority(weapon["kind"], close_laser)
        ):
            return i
    return None


class Resources:
    def __init__(self, config, arena):
        self.config = config
        self.arena = arena
        self.memory = {}
        self.failed = {}
        self.last_goal = None
        self.last_distance = math.inf
        self.progress_t = 0

    def update(self, obs):
        now = obs["t"]
        visible = set()
        for name in ("pickups", "crates", "cubes"):
            for item in obs[name]:
                key = item["id"]
                visible.add(key)
                self.memory[key] = name, item, now
        for event in obs["events"]:
            if event["type"] == "take_failed":
                if event["reason"] == "gone":
                    self.memory.pop(event["id"], None)
                else:
                    self.failed[event["id"]] = now + 0.8
            elif event["type"] == "took":
                self.memory.pop(event["id"], None)
        for key, (_, item, last) in list(self.memory.items()):
            if now - last > 16 or (
                key not in visible
                and distance(item["pos"], obs["self"]["pos"]) < 7
                and now - last > 0.4
            ):
                del self.memory[key]
        self.failed = {key: until for key, until in self.failed.items() if now < until}

    def take(self, obs, close_laser):
        me = obs["self"]
        reach = self.config["loot"]["pickup_radius"]
        options = []
        for item in obs["pickups"]:
            ds = distance(me["pos"], item["pos"])
            if (
                ds > reach
                or abs(me["pos"][2] - item["pos"][2]) > 4
                or item["id"] in self.failed
            ):
                continue
            slot = pickup_slot(me, item["kind"], close_laser)
            if slot is not None:
                options.append(
                    (weapon_priority(item["kind"], close_laser), -ds, item["id"], slot)
                )
        if not options:
            return {}
        _, _, object_id, slot = max(options)
        if me["weapons"][slot]:
            return {"take": object_id, "drop": slot}
        return {"take": object_id}

    def salvage(self, obs, threats, after_kill):
        """Collect nearby loose cubes when the current fight leaves room."""
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        reach = self.config["cubes"].get("pickup_radius", 3.5)
        limit = 26 if after_kill else 13
        options = []
        for cube in obs["cubes"]:
            point = cube["pos"]
            meters = distance(pos, point)
            if (
                meters > limit
                or abs(pos[2] - point[2]) > 3.8
                or zone_margin(point, obs["zone"], now, now + 3) < 1
                or not self.arena.walk_clear(pos, point, stats)
            ):
                continue
            if any(
                contact.exposed
                and distance(point, contact.actor["pos"]) < min(24, contact.meters - 6)
                for contact in threats
            ):
                continue
            cluster = sum(
                distance(point, other["pos"]) <= reach for other in obs["cubes"]
            )
            if me["cubes"] >= 10 and (not after_kill or cluster < 3) and meters > 7:
                continue
            options.append((min(6, cluster) * 4 - meters * 0.35, point))
        return tuple(max(options)[1][:2]) if options else None

    def objective(self, obs, threats, close_laser):
        me, now, zone = obs["self"], obs["t"], obs["zone"]
        pos = me["pos"]
        has_secondary = any(w and w["kind"] != "gun" for w in me["weapons"])
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        safe_farm = not any(
            c.actor["kind"] == "mech" and c.exposed and c.meters < 91 for c in threats
        )
        choices = []
        for key, (name, item, last) in self.memory.items():
            if key in self.failed:
                continue
            x, y, z = item["pos"]
            meters = distance(pos, item["pos"])
            if meters > 168 or abs(z - self.arena.ground(x, y)) > 3.8:
                continue
            eta = 1.3 * meters / me["stats"]["walk_speed"]
            if zone_margin(item["pos"], zone, now, now + eta + 4) < -3:
                continue
            farm = None
            if name == "pickups":
                if pickup_slot(me, item["kind"], close_laser) is None:
                    continue
                value = (21 if has_secondary else 42) + weapon_priority(
                    item["kind"], close_laser
                )
            elif name == "cubes":
                value = 43 if me["cubes"] < 10 else 20
                if (
                    meters < self.config["cubes"].get("pickup_radius", 3.5)
                    and abs(z - pos[2]) <= 4
                ):
                    value -= 18
            elif safe_farm and gun:
                value = 24 if me["cubes"] < 10 else 13
                value += 4 if not has_secondary else 0
                if now - last < 0.4:
                    farm = {
                        **item,
                        "kind": "crate",
                        "vel": (0, 0, 0),
                        "max_hp": self.config["crates"]["hp"],
                    }
            else:
                continue
            value -= 1.8 * eta + 0.45 * (now - last)
            if not safe_farm and meters > 28:
                value -= 14
            for threat in threats[:3]:
                if threat.actor["kind"] == "mech" and distance(
                    item["pos"], threat.actor["pos"]
                ) < min(44, threat.meters - 9):
                    value -= 27
            if key == self.last_goal:
                value += 3
            # For a reachable crate, position toward it until cubes can be collected.
            choices.append((value, key, (x, y), farm))
        if safe_farm and gun:
            for contact in threats:
                if (
                    contact.actor["kind"] != "npc"
                    or contact.meters > 65
                    or not contact.exposed
                ):
                    continue
                drone = contact.actor
                value = 30 - contact.meters * 0.25
                point = drone["pos"][:2]
                if zone_margin(point, zone, now, now + 6) > 3:
                    choices.append((value, drone["id"], tuple(point), drone))
        for drop in obs.get("drops", ()):
            pos_drop = drop["pos"]
            meters = distance(pos, pos_drop)
            eta = meters / me["stats"]["walk_speed"]
            if (
                meters > 130
                or drop["lands_in"] > min(11, eta + 5)
                or zone_margin(pos_drop, zone, now, now + eta + drop["capture_s"]) <= 0
            ):
                continue
            holder = drop["holder"]
            if holder not in (None, me["id"]):
                continue
            contest = any(
                c.actor["kind"] == "mech"
                and distance(c.actor["pos"], pos_drop) < max(38, meters - 5)
                for c in threats[:5]
            )
            if contest and holder != me["id"]:
                continue
            value = 50 + 20 * (1 - me["hp"] / me["max_hp"]) - 2.5 * eta
            if holder == me["id"]:
                value += 40
            choices.append((value, drop["id"], tuple(pos_drop[:2]), None))
        if not choices:
            return None, None
        value, key, goal, farm = max(choices, key=lambda choice: choice[0])
        if value < 2:
            return None, None
        meters = distance(pos, goal)
        if key != self.last_goal or meters < self.last_distance - 1 or meters <= 4:
            self.progress_t, self.last_distance = now, meters
        elif now - self.progress_t > 6:
            self.failed[key] = now + 9
            self.last_goal = None
            return None, None
        self.last_goal = key
        return goal, farm

    @staticmethod
    def upgrade(obs):
        offer = obs["offer"]
        if not offer:
            return {}
        me = obs["self"]
        values = {
            "nanites": 17,
            "plating": 15,
            "servos": 13,
            "armor": 11,
            "jets": 8,
            "gun_damage": 10,
            "gun_mag": 10,
            "laser_damage": 12,
            "laser_charge": 12,
            "lightning_efficiency": 13,
            "lightning_range": 14,
            "gun_velocity": 7,
            "gun_rate": 8,
            "gyros": 6,
            "capacitor": 5,
            "reactor": 7,
            "sensors": 7,
            "lightning_chain": 5,
            "laser_pierce": 4,
        }
        if me["hp"] <= 0.55 * me["max_hp"]:
            values["plating"] += 5
        if obs["alive"] <= 5 or obs["zone"]["next_radius"] <= 25:
            values["nanites"] -= 7
            values["armor"] += 4
        best = max(
            range(len(offer["choices"])),
            key=lambda i: values.get(offer["choices"][i]["id"], 0),
        )
        return {"pick": best}
