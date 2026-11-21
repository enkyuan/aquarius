"""Shorten direct approaches to a first weapon while preserving an escape reserve."""

import math
from heapq import nsmallest
from time import perf_counter

from .arena import TILE
from .movement import distance, zone_margin


class Opening:
    def __init__(self, arena, config):
        self.arena = arena
        self.config = config
        self.tick = 1 / config["match"]["tick_hz"]
        self.period = self.tick * config["match"]["decide_every"]
        self.alert_until = -math.inf
        self.wall_cells = {}
        # Index ground cells along obstacles. Confirm the actual prism, body size
        # and route on demand: the grid also marks boundaries and low overhangs.
        width = arena.width
        mask = arena.clearance
        for cell in range(width + 1, len(mask) - width - 1):
            if not mask[cell] & 1:
                continue
            for offset in (-width, width, -1, 1):
                if not mask[cell + offset] & 1:
                    x, y, _ = arena.point(cell)
                    tile = math.floor(x / TILE), math.floor(y / TILE)
                    self.wall_cells.setdefault(tile, []).append(cell)
                    break

    def observe(self, obs):
        # A sound has no actor id. When it could be a hunter, allow the same
        # quiet interval the server uses to retain a previously seen contact.
        memory = self.config["vision"]["memory_s"]
        if (
            obs["mechs"]
            or obs["projectiles"]
            or any(e["type"] == "damaged" for e in obs["events"])
        ):
            self.alert_until = obs["t"] + memory
        for sound in obs["sounds"]:
            if sound["kind"] not in ("death", "drop"):
                self.alert_until = max(
                    self.alert_until, obs["t"] - sound["age"] + memory
                )

    def lane(self, start, end, stats, deadline):
        """Check the sampled ground lane using the live slope and body limits."""
        length = distance(start, end)
        steps = max(1, math.ceil(length / 1.0))
        dz_max = min(
            stats["step_up"],
            length / steps * math.tan(math.radians(stats["max_slope_deg"])),
        )
        z = start[2]
        for i in range(1, steps + 1):
            if i % 8 == 0 and perf_counter() >= deadline:
                return False
            x = start[0] + (end[0] - start[0]) * i / steps
            y = start[1] + (end[1] - start[1]) * i / steps
            ground = self.arena.ground(x, y)
            if abs(ground - z) > dz_max + 1e-6 or not self.arena.body_clear(
                x, y, ground, stats["radius"], stats["height"]
            ):
                return False
            z = ground
        return True

    def wall_at(self, cell, stats):
        # Require a solid wall beside this cell over the full standing height.
        a = self.arena
        x, y, z = a.point(cell)
        tile = math.floor(x / TILE), math.floor(y / TILE)
        for other in (cell - a.width, cell + a.width, cell - 1, cell + 1):
            if a.clearance[other] & 1:
                continue
            bx, by, _ = a.point(other)
            across = (x + 2 * (bx - x), y + 2 * (by - y), z + stats["height"] / 2)
            for index in a.tiles.get(tile, ()):
                p = a.prisms[index]
                if (
                    p.base <= z + 0.05
                    and p.top >= z + stats["height"]
                    and p.intersects((x, y, across[2]), across)
                ):
                    return True
        return False

    def can_escape(self, stop, obs, energy, brake_time, deadline):
        a, me = self.arena, obs["self"]
        s = me["stats"]
        speed = s["boost_speed"]
        # From rest: accelerating all the way to boost speed plus running the
        # entire distance at that speed overestimates direct travel time.
        startup = speed / s["ground_accel"] + self.period
        radius = max(0, speed * (energy / s["boost_drain"] - startup))
        # An incomplete search simply declines the sprint.
        radius = min(radius, TILE * 2)
        x, y, _ = stop
        cells = (
            cell
            for tile in a.tiles_for(x - radius, y - radius, x + radius, y + radius)
            for cell in self.wall_cells.get(tile, ())
        )
        for meters, cell in nsmallest(
            16, ((distance(stop, a.point(c)), c) for c in cells)
        ):
            if meters > radius or perf_counter() >= deadline:
                return False
            p = a.point(cell)
            end = obs["t"] + self.period + brake_time + startup + meters / speed
            if (
                end > obs["zone"]["shrink_end"]
                or min(zone_margin(q, obs["zone"], obs["t"], end) for q in (p, stop))
                < 0
            ):
                continue
            if self.wall_at(cell, s) and self.lane(stop, p, s, deadline):
                return True
        return False

    def sprint(self, obs, item, deadline):
        me, now, zone = obs["self"], obs["t"], obs["zone"]
        if (
            now <= self.alert_until
            or not me["grounded"]
            or not me["in_zone"]
            or any(w and w["kind"] != "gun" for w in me["weapons"])
            or item["kind"] not in ("laser", "lightning")
            or perf_counter() >= deadline
        ):
            return None
        pos, goal, s = me["pos"], item["pos"], me["stats"]
        meters = distance(pos, goal)
        reach = self.config["loot"]["pickup_radius"]
        if meters <= reach or abs(goal[2] - self.arena.ground(*goal[:2])) > 4:
            return None
        speed, walk, accel = s["boost_speed"], s["walk_speed"], s["ground_accel"]
        energy = me["energy"] - self.period * s["boost_drain"]
        if (
            speed <= walk
            or min(walk, accel, energy, s["boost_drain"], s["energy_regen"]) <= 0
        ):
            return None
        # Can walking immediately after this command refill us by pickup?
        # Bound progress during the command and the excess coasting distance.
        max_speed = max(speed, math.hypot(*me["vel"][:2]))
        surplus = (max_speed - walk) ** 2 / (2 * accel) + self.tick * max_speed
        walk_time = (meters - reach - max_speed * self.period - surplus) / walk
        refill = (
            s["energy_regen_delay"] + (me["max_energy"] - energy) / s["energy_regen"]
        )
        if walk_time < refill:
            return None
        # Allow enough time even for an initial reversal and the equip cooldown.
        equip = max(
            self.config["loot"]["equip_s"],
            self.config["weapons"][item["kind"]]["cooldown"],
        )
        ready = now + meters / walk + 2 * max_speed / accel + self.period + equip
        if (
            ready > zone["shrink_end"]
            or min(zone_margin(p, zone, now, ready) for p in (pos, goal)) < 0
        ):
            return None
        if not self.lane(pos, goal, s, deadline):
            return None
        direction = (goal[0] - pos[0]) / meters, (goal[1] - pos[1]) / meters
        x, y, z = pos
        vx, vy = me["vel"][:2]
        for _ in range(self.config["match"]["decide_every"]):
            ax, ay = direction[0] * speed - vx, direction[1] * speed - vy
            f = min(1.0, accel * self.tick / max(math.hypot(ax, ay), 1e-9))
            vx, vy = vx + ax * f, vy + ay * f
            nxt = x + vx * self.tick, y + vy * self.tick
            if not self.lane((x, y, z), nxt, s, deadline):
                return None
            x, y = nxt
            z = self.arena.ground(x, y)
            if zone_margin(nxt, zone, now, ready) < 0:
                return None
        # Integrate the actual neutral command: the game accelerates toward
        # zero velocity before advancing position on each physics tick.
        halt_x, halt_y = x, y
        brake_ticks = math.ceil(math.hypot(vx, vy) / (accel * self.tick))
        speed_before_braking = math.hypot(vx, vy)
        for k in range(1, brake_ticks):
            remaining = 1 - k * accel * self.tick / speed_before_braking
            halt_x += vx * remaining * self.tick
            halt_y += vy * remaining * self.tick
        end = halt_x, halt_y
        brake_time = (
            math.ceil(brake_ticks / self.config["match"]["decide_every"]) * self.period
        )
        stop = *end, self.arena.ground(*end)
        if (
            not self.lane((x, y, z), stop, s, deadline)
            or not self.can_escape(stop, obs, energy, brake_time, deadline)
            or perf_counter() >= deadline
        ):
            return None
        return {"move": list(direction), "boost": True, "jump": False}
