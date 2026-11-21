"""Static geometry shared by the route planner and fire control."""

import gzip
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path
from struct import pack

TILE = 24.0
GRID = 3.0
NAV_CACHE = Path(__file__).resolve().parent.parent / "assets" / "nav-grid.gz"


def edges(flat):
    points = list(zip(flat[::2], flat[1::2]))
    return tuple((*a, *b) for a, b in zip(points, points[1:] + points[:1]))


def inside(x, y, ring):
    result = False
    for ax, ay, bx, by in ring:
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            result = not result
    return result


def edge_distance_sq(x, y, edge):
    ax, ay, bx, by = edge
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq:
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        ax += t * dx
        ay += t * dy
    return (x - ax) ** 2 + (y - ay) ** 2


class Prism:
    def __init__(self, data):
        ring = data["ring"]
        self.bounds = min(ring[::2]), min(ring[1::2]), max(ring[::2]), max(ring[1::2])
        self.base, self.top = data["base"], data["top"]
        self.outer = edges(ring)
        self.holes = tuple(edges(hole) for hole in data["holes"])
        self.edges = self.outer + tuple(edge for hole in self.holes for edge in hole)

    def contains(self, x, y):
        return inside(x, y, self.outer) and not any(
            inside(x, y, hole) for hole in self.holes
        )

    def distance_sq(self, x, y, limit):
        left, bottom, right, top = self.bounds
        dx, dy = max(left - x, 0.0, x - right), max(bottom - y, 0.0, y - top)
        if dx * dx + dy * dy > limit:
            return limit
        if dx == dy == 0 and self.contains(x, y):
            return 0.0
        return min(limit, min(edge_distance_sq(x, y, edge) for edge in self.edges))

    def intersects(self, a, b):
        """Clip the ray to the prism AABB, then test its projection against the rings."""
        lo, hi = 0.0, 1.0
        left, bottom, right, top = self.bounds
        for start, stop, low, high in zip(
            a, b, (left, bottom, self.base), (right, top, self.top)
        ):
            delta = stop - start
            if abs(delta) < 1e-9:
                if not low <= start <= high:
                    return False
                continue
            near, far = (low - start) / delta, (high - start) / delta
            lo, hi = max(lo, min(near, far)), min(hi, max(near, far))
            if lo > hi:
                return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        x, y = a[0] + lo * dx, a[1] + lo * dy
        if self.contains(x, y):
            return True
        for ax, ay, bx, by in self.edges:
            ex, ey = bx - ax, by - ay
            det = dx * ey - dy * ex
            if abs(det) < 1e-9:
                continue
            rx, ry = ax - a[0], ay - a[1]
            t = (rx * ey - ry * ex) / det
            u = (rx * dy - ry * dx) / det
            if lo <= t <= hi and 0 <= u <= 1:
                return True
        return self.contains(a[0] + hi * dx, a[1] + hi * dy)


class Arena:
    def __init__(self, map_dir, stats, cached_grid=NAV_CACHE):
        world_path = Path(map_dir) / "v2" / "world.json"
        world_bytes = world_path.read_bytes()
        world = json.loads(world_bytes)
        terrain = world["terrain"]
        terrain_bytes = (world_path.parent / terrain["file"]).read_bytes()
        self.terrain = array("h")
        self.terrain.frombytes(terrain_bytes)
        if sys.byteorder != "little":
            self.terrain.byteswap()
        self.tx, self.ty = terrain["minX"], terrain["minY"]
        self.nx, self.ny = terrain["nx"], terrain["ny"]
        self.step, self.scale = terrain["step"], terrain["scale"]
        bounds = world["bounds"]
        self.left, self.right = bounds["minX"], bounds["maxX"]
        self.bottom, self.top = bounds["minY"], bounds["maxY"]
        self.prisms = [Prism(data) for data in world["prisms"]]
        self.tiles = {}
        for index, prism in enumerate(self.prisms):
            left, bottom, right, top = prism.bounds
            # Include a margin so body checks usually need one tile lookup.
            for key in self.tiles_for(left - 8, bottom - 8, right + 8, top + 8):
                self.tiles.setdefault(key, []).append(index)

        self.width = int((self.right - self.left) / GRID)
        self.height = int((self.top - self.bottom) / GRID)
        self.heights = array("f")
        self.clearance = bytearray()
        radius, height = stats["radius"], stats["height"]
        fingerprint = hashlib.sha256(world_bytes)
        fingerprint.update(terrain_bytes)
        fingerprint.update(pack("<ddd", GRID, radius, height))
        self.grid_key = b"aquarius-nav-1\0" + fingerprint.digest()
        cells = self.width * self.height
        if cached_grid and Path(cached_grid).is_file():
            with gzip.open(cached_grid, "rb") as source:
                if source.read(len(self.grid_key)) == self.grid_key:
                    payload = source.read(cells * 5 + 1)
                    if len(payload) == cells * 5:
                        self.heights.frombytes(payload[: cells * 4])
                        if sys.byteorder != "little":
                            self.heights.byteswap()
                        self.clearance = bytearray(payload[cells * 4 :])
                        return
        self.build_grid(radius, height)

    def build_grid(self, radius, height):
        for j in range(self.height):
            y = self.bottom + (j + 0.5) * GRID
            for i in range(self.width):
                x = self.left + (i + 0.5) * GRID
                z = self.ground(x, y)
                self.heights.append(z)
                self.clearance.append(
                    self.body_clear(x, y, z, radius + 0.3, height)
                    + 2 * self.body_clear(x, y, z, radius * 1.5 + 0.3, height * 1.5)
                )

    def save_grid(self, destination):
        heights = array("f", self.heights)
        if sys.byteorder != "little":
            heights.byteswap()
        with (
            destination.open("wb") as target,
            gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as out,
        ):
            out.write(self.grid_key)
            out.write(heights)
            out.write(self.clearance)

    @staticmethod
    def tiles_for(left, bottom, right, top):
        for j in range(math.floor(bottom / TILE), math.floor(top / TILE) + 1):
            for i in range(math.floor(left / TILE), math.floor(right / TILE) + 1):
                yield i, j

    def ground(self, x, y):
        u = (x - self.tx) / self.step
        v = (y - self.ty) / self.step
        i = min(self.nx - 2, max(0, int(u)))
        j = min(self.ny - 2, max(0, int(v)))
        fx, fy = min(1, max(0, u - i)), min(1, max(0, v - j))
        k = j * self.nx + i
        data = self.terrain
        low = data[k] + fx * (data[k + 1] - data[k])
        high = data[k + self.nx] + fx * (data[k + self.nx + 1] - data[k + self.nx])
        return (low + fy * (high - low)) * self.scale

    def body_clear(self, x, y, z, radius, height):
        if not (
            self.left + radius <= x <= self.right - radius
            and self.bottom + radius <= y <= self.top - radius
        ):
            return False
        if radius <= 8:
            candidates = self.tiles.get(
                (math.floor(x / TILE), math.floor(y / TILE)), ()
            )
        else:
            candidates = {
                p
                for key in self.tiles_for(
                    x - radius, y - radius, x + radius, y + radius
                )
                for p in self.tiles.get(key, ())
            }
        for index in candidates:
            p = self.prisms[index]
            if p.top > z + 0.06 and p.base < z + height:
                limit = radius * radius
                if p.distance_sq(x, y, limit) < limit:
                    return False
        return True

    def walk_clear(self, a, b, stats):
        radius, height = stats["radius"], stats["height"]
        dx, dy = b[0] - a[0], b[1] - a[1]
        count = max(1, math.ceil(math.hypot(dx, dy) / 1.4))
        previous = a[2]
        for i in range(1, count + 1):
            x, y = a[0] + dx * i / count, a[1] + dy * i / count
            ground = self.ground(x, y)
            dz = ground - previous
            if dz > stats["step_up"] or not self.body_clear(
                x, y, ground, radius, height
            ):
                return False
            previous = ground
        return True

    def shot_clear(self, a, b):
        ax, ay, az = a
        bx, by, bz = b
        candidates = {
            index
            for key in self.tiles_for(
                min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)
            )
            for index in self.tiles.get(key, ())
        }
        if any(self.prisms[index].intersects(a, b) for index in candidates):
            return False
        count = max(1, math.ceil(math.hypot(bx - ax, by - ay) / 2.5))
        for i in range(1, count + 1):
            f = i / count
            if self.ground(ax + f * (bx - ax), ay + f * (by - ay)) >= az + f * (
                bz - az
            ):
                return False
        return True

    def cell(self, x, y):
        i = min(self.width - 1, max(0, int((x - self.left) / GRID)))
        j = min(self.height - 1, max(0, int((y - self.bottom) / GRID)))
        return j * self.width + i

    def point(self, cell):
        j, i = divmod(cell, self.width)
        return (
            self.left + (i + 0.5) * GRID,
            self.bottom + (j + 0.5) * GRID,
            self.heights[cell],
        )

    def nearby_cell(self, x, y, mask, rings=4, origin=None, stats=None):
        start = self.cell(x, y)
        if self.clearance[start] & mask and (
            origin is None or self.walk_clear(origin, self.point(start), stats)
        ):
            return start
        j, i = divmod(start, self.width)
        for r in range(1, rings + 1):
            candidates = []
            for row in range(max(0, j - r), min(self.height, j + r + 1)):
                for col in range(max(0, i - r), min(self.width, i + r + 1)):
                    cell = row * self.width + col
                    if not self.clearance[cell] & mask:
                        continue
                    point = self.point(cell)
                    if origin is None or self.walk_clear(origin, point, stats):
                        px, py, _ = point
                        candidates.append(((px - x) ** 2 + (py - y) ** 2, cell))
            if candidates:
                return min(candidates)[1]
        return None
