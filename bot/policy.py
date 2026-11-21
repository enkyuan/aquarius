"""Combine tactical goals, escape paths, and fire control into one action."""

import math
import random
from time import perf_counter

from .arena import Arena
from .combat import FireControl, angles, contacts
from .movement import (
    Router,
    cover_goal,
    distance,
    projectiles_near,
    rotate_goal,
    steer,
    zone_margin,
)
from .tactics import Resources, pickup_slot


class Bot:
    name = "aquarius"

    def __init__(self, hello):
        self.config = hello["config"]
        stats = self.config["mech"]
        self.arena = Arena(hello["map"]["dir"], stats)
        self.router = Router(self.arena, stats["radius"], stats["height"])
        self.fire_control = FireControl(self.config, self.arena)
        self.resources = Resources(self.config, self.arena)
        self.rng = random.Random(hello["seed"])
        self.damage = {}
        self.last_bearing = None
        self.retreating = False
        self.cover = None
        self.next_cover = 0
        self.explore = None
        self.explore_until = 0
        self.strafe = self.rng.choice((-1, 1))
        self.next_strafe = 0

    def update_damage(self, obs):
        now = obs["t"]
        self.damage = {
            i: hits
            for i, queue in self.damage.items()
            if (hits := [entry for entry in queue if now - entry[0] <= 4])
        }
        for event in obs["events"]:
            if event["type"] == "damaged" and event["by"] is not None:
                self.damage.setdefault(event["by"], []).append((now, event["amount"]))
                if event["bearing"] is not None:
                    self.last_bearing = now, event["bearing"]

    def plan_explore(self, obs):
        me, now = obs["self"], obs["t"]
        pos, zone, stats = me["pos"], obs["zone"], me["stats"]
        if (
            self.explore
            and now < self.explore_until
            and distance(pos, self.explore) > 7
            and zone_margin(self.explore, zone, now, now + 9) > 5
        ):
            return self.explore
        cx, cy = zone["next_center"]
        heading = math.atan2(cy - pos[1], cx - pos[0])
        # Look across new ground while gradually getting inside the next circle.
        spread = 1.4 if distance(pos, (cx, cy)) > zone["next_radius"] * 0.64 else 2.6
        options = []
        for offset in (0, -0.55, 0.55, -1.0, 1.0, -0.26, 0.26):
            yaw = heading + spread * offset
            x, y = pos[0] + 47 * math.cos(yaw), pos[1] + 47 * math.sin(yaw)
            z = self.arena.ground(x, y)
            margin = zone_margin((x, y), zone, now, now + 10)
            if margin < 4 or not self.arena.body_clear(
                x, y, z, stats["radius"] + 0.6, stats["height"]
            ):
                continue
            value = self.rng.uniform(0, 8) + min(30, margin) * 0.16
            if self.explore:
                value += min(65, distance((x, y), self.explore)) * 0.08
            near_x, near_y = pos[0] + 10 * math.cos(yaw), pos[1] + 10 * math.sin(yaw)
            if self.arena.walk_clear(pos, (near_x, near_y), stats):
                value += 10
            options.append((value, (x, y)))
        self.explore = max(options)[1] if options else tuple(zone["next_center"])
        self.explore_until = now + 7
        return self.explore

    def duel_goal(self, obs, target, kinds):
        pos = obs["self"]["pos"]
        now = obs["t"]
        enemy = target.actor
        ex, ey, _ = enemy["pos"]
        heading = math.atan2(ey - pos[1], ex - pos[0])
        if now >= self.next_strafe:
            self.strafe = self.rng.choice((-1, 1))
            self.next_strafe = now + self.rng.uniform(1.2, 2.3)
        enemy_weapons = enemy.get("weapons", ())
        if "lightning" in kinds and not "lightning" in enemy_weapons:
            preferred = 32
        elif "laser" in kinds or "lightning" in enemy_weapons:
            preferred = 78
        else:
            preferred = 54
        if enemy.get("hp", 1400) < 250 and "lightning" not in enemy_weapons:
            preferred -= 9
        forward = min(0.8, max(-0.85, (target.meters - preferred) / 40))
        side = self.strafe * math.sqrt(1 - forward * forward)
        dx = forward * math.cos(heading) - side * math.sin(heading)
        dy = forward * math.sin(heading) + side * math.cos(heading)
        return pos[0] + dx * 12, pos[1] + dy * 12

    def act(self, obs):
        started = perf_counter()
        me, now, zone = obs["self"], obs["t"], obs["zone"]
        pos, stats = me["pos"], me["stats"]
        self.update_damage(obs)
        self.fire_control.update(obs)
        self.resources.update(obs)
        nearby = contacts(obs, self.config, self.arena, self.damage, started + 0.004)
        hostiles = [c for c in nearby if c.actor["kind"] == "mech"]
        primary = hostiles[0] if hostiles else None
        active = [c for c in hostiles if c.exposed and c.pressure >= 2]
        health = me["hp"] / me["max_hp"]
        last_hit = self.last_bearing and now - self.last_bearing[0] < 2
        recent_damage = sum(
            amount
            for queue in self.damage.values()
            for t, amount in queue
            if now - t < 1.7
        )
        kinds = {w["kind"] for w in me["weapons"] if w}
        charging = any(
            w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in me["weapons"]
        )
        enemy_charge = any(c.actor.get("charging") and c.exposed for c in hostiles)
        close_laser = bool(
            primary
            and primary.meters < 47
            and "laser" in primary.actor.get("weapons", ())
            and me["energy"] > 32
        )
        early_bad_trade = bool(
            primary
            and primary.exposed
            and primary.meters > 57
            and primary.actor.get("charging")
            and "laser" not in kinds
            and (health < 0.91 or primary.meters > 110)
        )
        finishable = bool(
            primary
            and primary.exposed
            and primary.actor.get("hp", 1000) <= min(280, me["hp"] * 0.52)
        )
        crowded = len(active) >= 2 and (health < 0.88 or recent_damage > 120)
        should_leave = (
            crowded
            or early_bad_trade
            or (
                (health < 0.53 or (recent_damage > 240 and health < 0.74))
                and (active or last_hit)
                and not finishable
            )
        )
        if should_leave and not self.retreating:
            self.cover = None
            self.next_cover = 0
        if should_leave:
            self.retreating = True
        elif health > 0.88 or (finishable and not crowded):
            self.retreating = False
        rotate = rotate_goal(me, zone, now)
        darts = projectiles_near(obs, self.config)
        unsafe = not me["in_zone"] and zone["radius"] > 0
        quiet = self.retreating and not active and not last_hit and recent_damage == 0
        farm = None
        retreat_goal = None
        if self.retreating and (active or last_hit or darts):
            pursuers = [c.actor for c in (active or hostiles)[:2]]
            if not pursuers and self.last_bearing:
                yaw = self.last_bearing[1]
                pursuers = [
                    {
                        "pos": (
                            pos[0] + 40 * math.cos(yaw),
                            pos[1] + 40 * math.sin(yaw),
                            pos[2],
                        )
                    }
                ]
            if pursuers and (
                now >= self.next_cover
                or not self.cover
                or distance(pos, self.cover) < 2
            ):
                self.cover = cover_goal(
                    self.arena, me, pursuers, zone, now, started + 0.008, rotate
                )
                self.next_cover = now + 0.8
                if not self.cover:
                    tx, ty, _ = pursuers[0]["pos"]
                    length = max(1, distance(pos, (tx, ty)))
                    self.cover = (
                        pos[0] + 18 * (pos[0] - tx) / length,
                        pos[1] + 18 * (pos[1] - ty) / length,
                    )
            retreat_goal = self.cover
        take = self.resources.take(obs, close_laser)
        goal = None
        if unsafe:
            goal = rotate
        elif retreat_goal:
            goal = retreat_goal
        elif rotate:
            goal = rotate
        elif quiet:
            goal = tuple(pos[:2])
        else:
            resource, farm = self.resources.objective(obs, nearby, close_laser)
            priority_pickup = any(
                pickup_slot(me, item["kind"], close_laser) is not None
                and distance(pos, item["pos"]) < 42
                for item in obs["pickups"]
            )
            if (
                primary
                and primary.exposed
                and (primary.pressure >= 1.65 or primary.meters < 87)
            ):
                if (
                    resource
                    and priority_pickup
                    and not take
                    and not enemy_charge
                    and health > 0.73
                ):
                    goal = resource
                else:
                    goal, farm = self.duel_goal(obs, primary, kinds), None
            else:
                goal = resource
        if goal is None:
            goal = self.plan_explore(obs)
        if rotate and self.retreating and zone_margin(goal, zone, now, now + 3) < -2:
            goal = rotate
        waypoint = self.router.target(pos, goal, stats, started + 0.0115)
        fallback = me["aim"]
        if (primary and (self.retreating or primary.meters < 140)) or (
            self.retreating and last_hit
        ):
            if last_hit and (not primary or not primary.exposed):
                fallback = self.last_bearing[1], 0.0
            else:
                ex, ey, ez = primary.actor["pos"]
                fallback = angles(
                    (pos[0], pos[1], pos[2] + stats["muzzle_height"]), (ex, ey, ez + 5)
                )
        elif self.last_bearing and now - self.last_bearing[0] < 4:
            fallback = self.last_bearing[1], 0.0
        else:
            charge_warning = next(
                (
                    sound
                    for sound in obs["sounds"]
                    if sound["kind"] == "laser_charge" and sound["age"] < 0.2
                ),
                None,
            )
            if charge_warning:
                fallback = charge_warning["bearing"], 0.0
            elif distance(pos, waypoint) > 4:
                fallback = (
                    math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0]),
                    -0.035,
                )
        action = {"tick": obs["tick"], **take, **self.resources.upgrade(obs)}
        steer_threats = sorted(active, key=lambda c: not c.actor.get("charging", False))
        action.update(
            steer(
                self.arena,
                obs,
                waypoint,
                darts,
                [c.actor for c in steer_threats],
                self.retreating or unsafe,
                charging,
                started + 0.017,
            )
        )
        action.update(
            self.fire_control.decide(
                obs,
                nearby,
                farm,
                fallback,
                self.retreating,
                bool(quiet),
                started + 0.024,
            )
        )
        # Face within the server's pitch limits even while tracking an attacker above.
        action["aim"][1] = min(
            stats["max_pitch"], max(stats["min_pitch"], action["aim"][1])
        )
        return action
