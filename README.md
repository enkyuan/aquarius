# Aquarius

Aquarius is a Python bot for Builders Mech Royale. It routes around campus geometry, leads moving targets, fires the gun and secondary weapons through clear lanes, and uses hard cover to break contact. It runs with the standard library and reads the mounted map during the protocol handshake.

## Build and check

Extract the official kit beside this repository. From the repository root:

```sh
KIT=$(cd ../mechfight-kit && pwd -P)
docker build --platform linux/amd64 -t aquarius-bot .
python3 "$KIT/check/check.py" --map-dir /arena -- \
  docker run --platform linux/amd64 --rm -i --network none --cpus 1 --memory 512m \
  --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 \
  --mount "type=bind,src=$KIT/arena,dst=/arena,readonly" aquarius-bot
python3 -m dev.scenarios --kit "$KIT"
```

The official checker measures protocol compliance against a fixed recording. The scenario script checks targeted behaviors and approximates ground movement to exercise a route. Placement is measured by the official practice scorer.

`assets/nav-grid.gz` caches ground clearance for the kit map. Regenerate it after changes to the map or grid with `python3 -m dev.bake_nav --kit "$KIT"`. The bot checks it against the mounted map and recomputes the grid if the fingerprint changes.

## Policy

- `bot/policy.py` selects the current objective and combines it with a movement and firing action.
- `bot/arena.py` reads terrain and prisms, builds the ground grid, and handles collision and shot queries.
- `bot/movement.py` plans routes in bounded slices and steers around nearby obstacles, zone damage, and projectiles.
- `bot/combat.py` assesses threats and handles gun lead, laser charge and release, and lightning range.
- `bot/tactics.py` handles loot, supply drops, loadouts, and upgrade offers.

The bot uses ground routes and prioritizes gun plus laser. The gun handles firing opportunities during laser charge and cooldown, and the laser takes priority when ready to release. Nearby cubes are a priority after a kill. Retreating suspends gunfire to give repairs a chance to start. Its alternate loadout uses lightning for close-range fights.

`bot/mechfight.py` is the Python protocol wrapper supplied by the official [Mech Royale starter kit](https://www.joinbuilders.org/buildathon/tracks/builders).
