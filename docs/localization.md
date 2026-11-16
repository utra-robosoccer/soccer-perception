# Localization

The localizer estimates the robot's pose on the field, `(x, y, yaw)`, with a particle filter
(Monte Carlo localization). This page explains what it does and why, for someone who has not
written one before. The code is in `localization/`; start with `particle_filter.py`.

## The idea

We cannot compute the pose directly from one camera frame: a few line segments and a T junction
look the same from many places. Instead the filter keeps many guesses at once. Each guess is a
**particle**: one pose `(x, y, yaw)` with a **weight** saying how believable it is. Together the
particles approximate "where might the robot be, and how likely is each place".

Every cycle:

1. **Predict** (`predict`). Odometry says "I moved 3 cm forward and turned 1 deg". Every
   particle is moved by that amount plus its own random error. The cloud spreads by as much as
   odometry cannot be trusted.
2. **Update** (`update`). For each particle: if the robot were here, how well would the things
   it sees match the map? The particle's weight is multiplied by that score.
3. **Resample**, only when needed. If a few particles hold most of the weight, draw a new set
   in proportion to weight: good guesses are copied, bad ones disappear.
4. **Estimate** (`estimate`). Report the dominant cluster of particles as the pose.

`ParticleFilter.step(delta, frame)` runs one cycle.

## Inputs

Only two types enter the filter after initialization:

- `OdometryDelta(dx, dy, dyaw, covariance, timestamp, dt)`: relative motion in the robot frame.
- `ObservationFrame`: detections in the robot frame.
  - `LinePointsObservation`: points believed to lie on some field line.
  - `LandmarkObservation`: a class (`L`, `T`, `X`, `goal_post`, `penalty_mark`,
    `center_mark`), a position relative to the robot, and range / bearing standard deviations.

Neither says where on the field anything is, or which landmark was seen.

## Initialization

- `init_gaussian(pose, position_std, yaw_std)`: an approximately known pose.
- `init_uniform()`: no idea at all; uniform over the carpet and all headings.
- `init_uniform(x_range=...)`: a partial prior, for example "somewhere in our own half".
- Calling any of them again reinitializes the filter.

## Motion model (`motion_model.py`)

Each particle gets `delta + noise`, with

```
translation std = translation_noise * distance + translation_from_rotation * |turn|   (floor: min_translation_std)
rotation std    = rotation_noise * |turn| + rotation_from_translation * distance      (floor: min_rotation_std)
```

The floors keep the cloud from collapsing to a single point while standing still. If an
odometry message is missing, call `predict(None)`: the particles only diffuse, by more.
The filter's noise is deliberately larger than the simulated odometry's: it must also cover
drift it cannot see.

## Measurement model (`measurement_model.py`)

**Line points.** Each point is transformed into the field frame using the particle's pose and
looked up in the distance map. With `d` the distance to the nearest line and `sigma` the point's
uncertainty:

```
score = floor + (1 - floor) * exp(-0.5 * (d / sigma)^2)
```

**Landmarks.** For an observed landmark of class `c`, the model computes where every map
landmark of class `c` would appear from the particle, and compares each with the observation
(error along the viewing direction scaled by the range std, across it by range x bearing std).
Because the identity is unknown:

- `correspondence: nearest` uses the best-matching candidate (default);
- `correspondence: mixture` adds up all candidates.

The same `floor + (1 - floor) * fit` form is used.

**Why the floor matters.** Without it, one false detection would multiply a correct particle's
weight by almost zero. With it, a detection that fits nothing costs at most a fixed factor
(`log(0.05)` for a landmark), so outliers and false positives slow the filter down but cannot
veto the truth.

**Why line points are down-weighted.** Forty points along the same line are not forty
independent measurements. `line_point_weight: 0.3` raises each point's score to the power 0.3
so that lines do not overwhelm everything else and collapse the cloud.

Particles outside the carpet get a large penalty.

## Resampling

The effective sample size `ESS = 1 / sum(w^2)` is N when all weights are equal and 1 when one
particle holds everything. The filter resamples when `ESS < resample_ess_fraction * N` (default
half), using systematic (low-variance) resampling: one random offset, then N evenly spaced
picks along the cumulative weights. In the benchmarks this happens on 17 to 44 % of the
updates, depending on the scenario, not on every one.

## Pose estimate (`estimation.py`)

The weighted mean of all particles is wrong whenever there is more than one cluster: two
mirror-image clusters average to the centre of the field. The estimator instead:

1. bins particles into a coarse histogram over `(x, y, yaw)` (0.4 m, 30 deg; yaw wraps);
2. finds the heaviest 3x3x3 neighbourhood;
3. reports the weighted mean of the particles in it (circular mean for yaw), their covariance,
   and their share of the total weight;
4. removes them and repeats, to list the other modes.

`PoseEstimate` contains:

| Field | Meaning |
|---|---|
| `pose`, `x`, `y`, `yaw` | the dominant mode |
| `covariance`, `position_std`, `yaw_std` | spread of that mode |
| `mode_weight` | share of the belief in that mode |
| `confidence` | `mode_weight` discounted by the mode's spread, in [0, 1] |
| `converged` | mode weight >= 0.7, position std <= 0.25 m, yaw std <= 10 deg, and not ambiguous |
| `symmetric_ambiguity` | another mode sits at the mirrored pose |
| `modes` | every mode found, as alternative hypotheses |

`converged` and `confidence` are the filter's own opinion. They measure how concentrated the
particles are, not whether they are right. See "Limits" below.

## Field symmetry

A RoboCup field is unchanged by a half turn about its centre. A robot at `(x, y, yaw)` and one
at `(-x, -y, yaw + 180 deg)` see exactly the same thing, forever. No observation can tell them
apart; only outside information can (which half the robot started in, the game state, a
teammate).

The filter is built around that fact:

- **No side information** (`init_uniform()`): the true posterior is exactly symmetric, so the
  filter keeps it symmetric (`maintain_symmetry`). It reports both poses in `modes`, sets
  `symmetric_ambiguity`, and does not claim convergence. Without this, resampling noise picks
  a side at random and the filter becomes confident about a coin flip.
- **Outside information arrives**: call `resolve_symmetry(reference_xy)` with any point on the
  robot's side of the field centre.
- **Side information at start** (`init_gaussian`, or `init_uniform` with a range): the filter
  remembers an **anchor**, the last place it had reason to believe in. The anchor follows the
  estimate once the filter has stayed converged for `anchor_hold_updates` updates.

## Recovery from large errors and kidnapping

If the robot is picked up and moved, odometry reports nothing and every particle is wrong. The
filter notices through **agreement**: the average per-observation score under the current
belief. It tracks a slow and a fast running average; when the fast one falls below the slow one
by more than `injection_trigger`, it forces a resampling and replaces up to
`max_injection_fraction` of the particles with fresh hypotheses. A small share (1 %) is also
replaced at every ordinary resampling.

Fresh hypotheses are never taken from ground truth. They come from two sources:

- **Sensor resetting** (70 %): pick an observed landmark, pick a map landmark of the same
  class, and place the robot at the observed range from it, with the heading that puts it at
  the observed bearing. Such a pose explains at least one observation.
- **Uniform** (30 %): anywhere on the carpet.

Two safeguards, both found necessary in the benchmarks:

- **Low starting weight** (`injected_particle_weight: 0.1`). A view with few landmarks is
  aliased: a grid of perpendicular lines looks the same after a quarter turn. A fresh
  hypothesis at such a pose can explain one frame as well as the truth does. Starting it at a
  tenth of a survivor's weight means it must be better for several frames before it takes over.
  After a real kidnapping the survivors explain nothing, and it takes over within a few frames.
- **Same-side prior** (`symmetry_prior: same_side`). A recovered pose and its mirror image are
  equally good. Each fresh hypothesis is placed on the side of the field centre nearer the
  anchor: "the robot was more likely moved a short way than across the field". If it really
  was carried across the centre, the filter recovers the **mirrored** pose and cannot know.
  Benchmark scenario `s10b` shows exactly this failure.

## Configuration

`configs/localization/particle_filter.yaml` holds every parameter with a comment.
`ParticleFilterConfig.load(overrides={...})` applies nested overrides.

## Limits

- **Symmetry** cannot be resolved from the field alone (above).
- **`converged` can be wrong.** It is true whenever the particles are concentrated. A filter
  started at the mirrored pose reports convergence throughout (scenario `s11c`), and a filter
  started from a partial prior can briefly report it on a look-alike pose. The benchmark's
  `overconfident_fraction` measures this.
- **Negative information is unused.** The model scores what was seen, not what should have
  been seen and was not.
- **No landmark orientation.** A T junction's direction is ignored; using it would remove much
  of the aliasing.
- **Detector confidence is ignored.** `LandmarkObservation.confidence` is carried but unused.
- **Fixed particle count.** 1000 particles for every field; no KLD-style adaptation.
- **Flat ground, upright camera model.** Observations are assumed already projected onto the
  ground plane in the robot frame.
- **Parameters were tuned on benchmark seeds 0-4** (see [validation.md](validation.md)).
