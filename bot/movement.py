"""Incremental ground routing and acceleration-aware local steering."""

import math
from heapq import heappop, heappush
from time import perf_counter

from .arena import GRID


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def circle_at(zone, now, when):
    start = max(now, zone["shrink_start"])
    end = zone["shrink_end"]
    if end <= start:
        return (*zone["next_center"], zone["next_radius"])
    f = min(1.0, max(0.0, (when - start) / (end - start)))
    x, y = zone["center"]
    nx, ny = zone["next_center"]
    return (
        x + f * (nx - x),
        y + f * (ny - y),
        zone["radius"] + f * (zone["next_radius"] - zone["radius"]),
    )


def zone_margin(point, zone, now, when):
    x, y, radius = circle_at(zone, now, when)
    return radius - math.hypot(point[0] - x, point[1] - y)


def rotate_goal(me, zone, now):
    pos = me["pos"]
    nx, ny = zone["next_center"]
    next_radius = zone["next_radius"]
    distance_to_center = distance(pos, (nx, ny))
    # Pick an interior point along the radial entry, with a smaller buffer late.
    safe_radius = max(0.0, next_radius - min(33, next_radius * 0.35))
    travel = max(0.0, distance_to_center - safe_radius)
    eta = 5 + 1.35 * travel / me["stats"]["walk_speed"]
    needs_rotation = (
        (not me["in_zone"] and zone["radius"] > 0)
        or (travel > 1 and now + eta >= zone["shrink_end"])
        or zone_margin(pos, zone, now, now + 3) < -1
    )
    if needs_rotation and distance_to_center > 0.1:
        ratio = min(1.0, safe_radius / distance_to_center)
        return nx + ratio * (pos[0] - nx), ny + ratio * (pos[1] - ny)
    return None


class Router:
    def __init__(self, arena, base_radius, base_height):
        self.arena = arena
        self.base_radius = base_radius
        self.base_height = base_height
        self.key = None
        self.frontier = []
        self.parent = {}
        self.cost = {}
        self.path = []
        self.path_cursor = 0
        self.edges = {}
        self.edge_shape = None
        self.visited = 0
        self.failed = False

    def target(self, pos, goal, stats, deadline):
        a = self.arena
        meters = distance(pos, goal)
        if (
            meters < 26
            and abs(pos[2] - a.ground(*pos[:2])) < 1.6
            and a.walk_clear(pos, goal, stats)
        ):
            return goal
        mask = (
            1
            if stats["radius"] <= self.base_radius + 0.35
            and stats["height"] <= self.base_height
            else 2
        )
        old_goal = self.key[0] if self.key else None
        if (
            old_goal is not None
            and distance(a.point(old_goal), goal) < 7
            and self.key[1] == mask
        ):
            dest = old_goal
        else:
            dest = a.nearby_cell(*goal[:2], mask)
        if dest is None:
            return goal
        shape = stats["radius"], stats["height"], stats["step_up"]
        if self.edge_shape != shape:
            self.edges.clear()
            self.edge_shape = shape
        key = dest, mask, shape
        rejoin = (
            self.failed
            and distance(pos, a.point(self.start)) > 13
            and (not self.path or distance(pos, a.point(self.path[-1])) < 5)
        )
        if self.key != key or rejoin:
            if len(self.edges) > 40000:
                self.edges.clear()
            start = a.nearby_cell(*pos[:2], mask, rings=2, origin=pos, stats=stats)
            if start is None:
                return goal
            self.key, self.start = key, start
            self.frontier = [(0.0, 0.0, start)]
            self.parent = {start: None}
            self.cost = {start: 0.0}
            self.path = []
            self.path_cursor = 0
            self.visited = 0
            self.failed = False
            self.best = distance(a.point(start), a.point(dest)), start
        if not self.path and not self.failed:
            self.search(dest, mask, stats, deadline)
        if not self.path:
            return goal
        # Use the furthest nearby waypoint with a direct body-clear approach.
        last = min(len(self.path) - 1, self.path_cursor + 6)
        first = max(0, self.path_cursor - 2)
        for i in range(last, first - 1, -1):
            waypoint = a.point(self.path[i])
            if a.walk_clear(pos, waypoint, stats):
                if i > self.path_cursor and distance(pos, waypoint) < 8:
                    self.path_cursor = i
                return waypoint[:2]
        return a.point(self.path[self.path_cursor])[:2]

    def search(self, dest, mask, stats, deadline):
        a = self.arena
        width, height = a.width, a.height
        goal_row, goal_col = divmod(dest, width)
        heights, clearance = a.heights, a.clearance
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        for i in range(320):
            if not self.frontier or self.visited >= 26000:
                self.failed = True
                if self.best[1] != self.start:
                    self.finish(self.best[1])
                else:
                    self.frontier.clear()
                return
            if i % 4 == 0 and perf_counter() >= deadline:
                return
            _, g, cell = heappop(self.frontier)
            if g != self.cost[cell]:
                continue
            if cell == dest:
                self.finish(cell)
                return
            self.visited += 1
            row, col = divmod(cell, width)
            remaining = math.hypot(row - goal_row, col - goal_col) * GRID
            if remaining < self.best[0]:
                self.best = remaining, cell
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                other = nr * width + nc
                diagonal = dr and dc
                if not clearance[other] & mask:
                    continue
                if diagonal and not (
                    clearance[cell + dc] & clearance[cell + dr * width] & mask
                ):
                    continue
                climb = heights[other] - heights[cell]
                if climb > stats["step_up"] * (2.4 if diagonal else 1.9):
                    continue
                cost = g + (1.414 if diagonal else 1) + abs(climb) * 0.10
                if cost < self.cost.get(other, math.inf):
                    key = cell, other
                    clear = self.edges.get(key)
                    if clear is None:
                        clear = a.walk_clear(a.point(cell), a.point(other), stats)
                        self.edges[key] = clear
                    if not clear:
                        continue
                    self.parent[other] = cell
                    self.cost[other] = cost
                    heuristic = math.hypot(nr - goal_row, nc - goal_col)
                    heappush(self.frontier, (cost + 1.35 * heuristic, cost, other))

    def finish(self, cell):
        path = []
        while cell is not None:
            path.append(cell)
            cell = self.parent[cell]
        self.path = path[::-1]
        self.frontier.clear()


def projectiles_near(obs, config):
    x, y, _ = obs["self"]["pos"]
    relevant = []
    for p in obs["projectiles"]:
        px, py, _ = p["pos"]
        vx, vy, _ = p["vel"]
        d = math.hypot(px - x, py - y)
        if p["kind"] == "mortar":
            if d <= 75:
                relevant.append((d, p, config["weapons"]["mortar"]["gravity"]))
        elif d <= 160 and (x - px) * vx + (y - py) * vy > -20:
            relevant.append((d, p, 0.0))
    return [
        (p, gravity) for _, p, gravity in sorted(relevant, key=lambda item: item[0])[:7]
    ]


def projectile_risk(a, b, t, dt, bullets, radius, height):
    risk = 0.0
    for p, gravity in bullets:
        px, py, pz = p["pos"]
        vx, vy, vz = p["vel"]
        ox, oy = px + vx * t - a[0], py + vy * t - a[1]
        dx, dy = vx * dt - (b[0] - a[0]), vy * dt - (b[1] - a[1])
        q = dx * dx + dy * dy
        f = min(1.0, max(0.0, -(ox * dx + oy * dy) / q)) if q else 0.0
        dist_sq = (ox + dx * f) ** 2 + (oy + dy * f) ** 2
        margin = radius + (7.5 if gravity else 0.75)
        if dist_sq >= margin * margin:
            continue
        impact = t + dt * f
        dz = pz + vz * impact - 0.5 * gravity * impact**2 - (a[2] + (b[2] - a[2]) * f)
        if (-5 if gravity else -0.4) <= dz <= height + (7 if gravity else 0.4):
            risk += (11 if gravity else 5) * (1 - 0.45 * dist_sq / (margin * margin))
    return risk


def cover_goal(arena, me, threats, zone, now, deadline, preferred=None):
    pos, stats = me["pos"], me["stats"]
    primary = threats[0]["pos"]
    away = math.atan2(pos[1] - primary[1], pos[0] - primary[0])
    best, value = None, -math.inf
    for length in (10, 19, 31):
        for offset in (0, 45, -45, 90, -90, 135, -135, 180):
            if perf_counter() >= deadline:
                return best
            yaw = away + math.radians(offset)
            x, y = pos[0] + math.cos(yaw) * length, pos[1] + math.sin(yaw) * length
            z = arena.ground(x, y)
            if not arena.body_clear(x, y, z, stats["radius"] + 0.4, stats["height"]):
                continue
            point = x, y, z
            t = now + 2 + length / stats["walk_speed"]
            inside = zone_margin(point, zone, now, t)
            if inside < -7:
                continue
            score = (
                min(0.0, inside)
                - length * 0.25
                + 0.10 * (distance(point, primary) - distance(pos, primary))
            )
            for i, enemy in enumerate(threats[:2]):
                ex, ey, ez = enemy["pos"]
                muzzle = ex, ey, ez + 7 * (1 + 0.05 * enemy.get("cubes", 0))
                if not arena.shot_clear(muzzle, (x, y, z + stats["height"] * 0.5)):
                    score += 24 if i == 0 else 12
            if arena.walk_clear(pos, point, stats):
                score += 8
            if preferred:
                score -= 0.16 * distance(point, preferred)
            if score > value:
                best, value = point[:2], score
    return best


def steer(arena, obs, goal, bullets, threats, retreat, charging, deadline):
    me, zone, now = obs["self"], obs["zone"], obs["t"]
    pos, vel, stats = me["pos"], me["vel"], me["stats"]
    speed = stats["walk_speed"]
    radius, height = stats["radius"], stats["height"]
    walk_goal = distance(pos, goal)
    heading = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
    enemy = threats[0] if threats else None
    laser_warning = bool(enemy and enemy.get("charging"))
    if walk_goal < 3 and bullets:
        vx, vy, _ = bullets[0][0]["vel"]
        heading = math.atan2(vy, vx) + math.pi / 2
    reserve = 10 if retreat or not me["in_zone"] else 32
    can_boost = me["grounded"] and me["energy"] > reserve
    if retreat or bullets:
        offsets = (0, 30, -30, 65, -65, 105, -105, 150, -150, 180)
    else:
        offsets = (0, 28, -28, 62, -62, 100, -100, 145, -145, 180)
    throttle = 1.0 if bullets or laser_warning else min(1.0, walk_goal / 3.0)
    directions = [
        (
            math.cos(heading + math.radians(d)) * throttle,
            math.sin(heading + math.radians(d)) * throttle,
        )
        for d in offsets
    ]
    candidates = [(0.0, 0.0, False, False)]
    for i, (dx, dy) in enumerate(directions):
        if can_boost and (retreat or bullets or not me["in_zone"]) and i < 7:
            candidates.append((dx, dy, True, False))
        candidates.append((dx, dy, False, False))
    if me["grounded"] and walk_goal > 3:
        candidates.extend((dx, dy, False, True) for dx, dy in directions[:3])
    step, count = 0.12, 5
    best, best_score = candidates[0], -math.inf
    for dx, dy, boost, jump in candidates:
        if perf_counter() >= deadline:
            break
        x, y, z = pos
        vx, vy, vz = vel
        grounded = me["grounded"] and not jump
        if jump:
            vz = stats["jump_speed"]
        target_speed = speed if not boost else stats["boost_speed"]
        if charging:
            target_speed *= stats["charge_move_mult"]
        risk, contacts = 0.0, 0
        for i in range(count):
            prev = x, y, z
            ax, ay = dx * target_speed - vx, dy * target_speed - vy
            accel = stats["ground_accel"] if grounded else stats["air_accel"]
            factor = min(1.0, accel * step / max(math.hypot(ax, ay), 0.001))
            vx, vy = vx + ax * factor, vy + ay * factor
            nx, ny = x + vx * step, y + vy * step
            ground = arena.ground(nx, ny)
            if grounded:
                nz = ground
                passable = ground - z <= min(
                    stats["step_up"],
                    math.tan(math.radians(stats["max_slope_deg"]))
                    * distance(prev, (nx, ny))
                    + 0.05,
                )
                if ground < z - 0.65:
                    nz, grounded = z, False
            else:
                vz -= stats["gravity"] * step
                nz = max(ground, z + vz * step)
                passable = ground <= z + max(0, vz * step) + 0.3
                if nz == ground and passable:
                    grounded, vz = True, 0
            if passable and arena.body_clear(nx, ny, nz, radius, height):
                x, y, z = nx, ny, nz
            else:
                contacts += 1
                # Preserve velocity parallel to the obstacle on a plausible slide.
                if abs(vx) >= abs(vy):
                    axes = ((nx, y, True), (x, ny, False))
                else:
                    axes = ((x, ny, False), (nx, y, True))
                for sx, sy, along_x in axes:
                    sz = arena.ground(sx, sy) if grounded else z
                    if abs(sz - z) <= stats["step_up"] and arena.body_clear(
                        sx, sy, sz, radius, height
                    ):
                        x, y, z = sx, sy, sz
                        vx, vy = (vx, 0) if along_x else (0, vy)
                        break
                else:
                    vx, vy = 0, 0
            risk += projectile_risk(
                prev, (x, y, z), i * step, step, bullets, radius, height
            )
        dest = x, y, z
        remaining = distance(dest, goal)
        score = (walk_goal - remaining) * 1.18 - contacts * 1.25 - risk
        if walk_goal < 3:
            score -= 0.37 * math.hypot(vx, vy)
        if boost:
            score -= 1.3 if retreat else 2.2
        if jump:
            score -= 3.6
        margin = zone_margin(dest, zone, now, now + count * step)
        if margin < 0:
            score += min(0.0, margin - min(0.0, zone_margin(pos, zone, now, now))) * (
                zone["dps"] / 35 + 1
            )
        if enemy and (retreat or laser_warning) and contacts < 3:
            ex, ey, ez = enemy["pos"]
            origin = ex, ey, ez + 7 * (1 + 0.05 * enemy.get("cubes", 0))
            if not arena.shot_clear(origin, (x, y, z + height * 0.5)):
                score += 10
            if enemy.get("charging"):
                yaw = math.atan2(y - ey, x - ex)
                delta = abs(math.remainder(yaw - enemy["aim"][0], math.tau))
                score += min(3, delta * 18)
        if score > best_score:
            best, best_score = (dx, dy, boost, jump), score
    return {"move": list(best[:2]), "boost": best[2], "jump": best[3]}
