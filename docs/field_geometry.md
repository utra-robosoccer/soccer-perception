# Field geometry

`geometry/field.py` is the only place that knows what a field looks like. It reads one of
`configs/fields/{s,m,l}_field.yaml` and produces painted primitives, goals and extents in the
field frame. To use another field size, add a YAML file; no code changes.

## Configuration

Dimensions come from the RoboCup Humanoid Soccer League 2026 rulebook (dated 2026-06-29), as
supplied with the task:

| | S | M | L |
|---|---|---|---|
| Length x width | 9.0 x 6.0 | 14.0 x 9.0 | 22.0 x 14.0 |
| Line width | 0.05 | 0.05 | 0.12 |
| Goal area (depth x width) | 1.0 x 3.0 | 1.0 x 4.0 | 1.0 x 5.0 |
| Penalty area (depth x width) | 2.0 x 4.0 | 3.0 x 6.0 | 3.5 x 7.0 |
| Penalty mark distance | 1.5 | 2.0 | 2.5 |
| Centre circle diameter | 1.5 | 3.0 | 4.0 |
| Corner arc radius | none | 0.5 | 1.0 |
| Border strip (minimum) | 1.0 | 1.0 | 1.0 |
| Goal width, allowed | 1.8-1.9 | 2.4-2.6 | 2.6-3.1 |
| Goal depth, allowed | 0.5-1.0 | 0.7-1.2 | 1.0-2.0 |

Choices made where the rulebook gives a range, and assumptions where the task gave nothing:

| Value | S | M | L | Status |
|---|---|---|---|---|
| Goal width (inner distance between posts) | 1.8 | 2.6 | 3.0 | chosen within the range |
| Goal depth | 0.6 | 0.8 | 1.2 | chosen within the range |
| Goal height | 1.2 | 1.8 | 1.8 | **assumption** |
| Post diameter | 0.10 | 0.10 | 0.12 | **assumption** |
| Mark size (arm length of the crosses) | 0.15 | 0.15 | 0.30 | **assumption** |
| Posts centred on the goal line's centreline | | | | **assumption** |
| Ball radius 0.07 m (`configs/scenes/default.yaml`) | | | | **assumption** |

Check the assumptions against the rulebook before relying on goal or ball appearance. None of
them affects line geometry.

## Outside-of-line measurements

The rulebook measures every dimension to the **outside** of the lines. Lines are stored as a
centreline plus a width, so each centreline is shifted inward by half a line width:

| Feature | Centreline |
|---|---|
| Touchlines | `y = +-(width/2 - w/2)` |
| Goal lines | `x = +-(length/2 - w/2)` |
| Halfway line | `x = 0` (centred) |
| Goal / penalty area front line | `x = +-(length/2 - depth + w/2)` |
| Goal / penalty area side lines | `y = +-(area_width/2 - w/2)` |
| Centre circle | radius `diameter/2 - w/2` |
| Corner arcs | radius `r - w/2`, centred on the outer corner |
| Penalty mark | centre at `x = +-(length/2 - distance)` |

For the S field (w = 0.05) the boundary centrelines are at x = ±4.475 and y = ±2.975, and
the outer paint edge is exactly at ±4.5 and ±3.0. `tests/test_field_geometry.py` checks
both edges of every line type on all three fields.

Straight lines are rectangles with square ends; arcs are annular sectors.
`FieldGeometry.distance_to_paint` is exact for both.

## Landmarks

`geometry/landmarks.py` derives point landmarks from the primitives (junctions are computed as
intersections of centrelines, so they follow the convention above automatically).

| Class (`LandmarkType`) | Count | What it is |
|---|---|---|
| `L` | 12 | corners: 4 field, 4 goal area, 4 penalty area |
| `T` | 10 | halfway line x touchlines (2), goal/penalty area side lines x goal line (8) |
| `X` | 2 | halfway line x centre circle |
| `goal_post` | 4 | post bases |
| `penalty_mark` | 2 | |
| `center_mark` | 1 | centre of the centre circle |

Three things are kept separate:

- **Geometry**: where paint and posts are.
- **Identity**: `Landmark.id`, for example `T:penalty_area:pos_x:neg_y`. Only the map and the
  evaluation code use identities.
- **Class**: what a detector can report. Every class is ambiguous: there are ten T junctions,
  and the whole field maps onto itself under a half turn about its centre, so even the single
  centre mark does not tell you which way you are facing.

Observations carry a class and a relative position, never an identity. Corner-arc junctions
are not landmarks (they add no class a detector could reliably separate).

## Distance map

`geometry/distance_map.py` stores the distance from each ground point to the nearest paint on a
regular grid over the carpet (field plus border strip), as float32 metres at grid nodes, with
bilinear lookup. Outside the grid the value keeps growing with distance.

| Field | Grid (2.5 cm) | Memory |
|---|---|---|
| S | 321 x 441 | 0.57 MB |
| M | 441 x 641 | 1.13 MB |
| L | 641 x 997 | 2.56 MB |

Maps are computed on first use (well under a second) and cached in memory under the field's
content hash, so any change to a field's dimensions yields a new map automatically. Lookup
error against the exact distance stays below 0.75 of a cell (tested on 20 000 random points per
field).

## Extents

`FieldGeometry.half_extent` is the half-size of the carpet: the field plus the border strip. In
x the border is widened when needed so that the goals stand on the carpet (the L field's 1.2 m
goal depth exceeds the 1.0 m minimum border). Particles are confined to the carpet.
