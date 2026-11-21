"""Build the portable, map-fingerprinted navigation cache from the official kit."""

import argparse
import gzip
import json
from pathlib import Path

from bot.arena import NAV_CACHE, Arena


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.kit / "check" / "stream.jsonl.gz", "rt") as stream:
        stats = json.loads(next(stream))["config"]["mech"]
    arena = Arena(args.kit / "arena", stats, cached_grid=None)
    NAV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    arena.save_grid(NAV_CACHE)
    print(
        f"Baked {len(arena.clearance):,} cells into {NAV_CACHE.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
