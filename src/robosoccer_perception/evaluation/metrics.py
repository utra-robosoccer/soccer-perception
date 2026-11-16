"""Localization metrics. Ground truth is used here, and only here and in the plots.

Definitions (thresholds in `MetricThresholds`)
----------------------------------------------
localized       position error < 0.30 m and |yaw error| < 10 deg at that step.
convergence     the first time from which the run stays localized for 2 s without a break.
tracking window every step after the first convergence, except the recovery windows that
                follow a kidnapping (from the kidnap until the next convergence). RMSE, MAE and
                the 95th percentiles are computed over this window; `*_all` variants cover
                every step of the run.
diverged        position error > 1.0 m or |yaw error| > 30 deg. `diverged_fraction` is the
                share of the tracking window spent diverged; `final_diverged` says whether the
                run ended not localized.
wrong half      the estimate is "localized" with respect to the half-turn MIRROR of the true
                pose but not the true pose itself: the symmetry failure.
overconfident   the filter reports `converged` while the true error says it is not localized.

Yaw errors are circular differences. Positions are in metres, angles in degrees.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.localization.estimation import mirror_pose


@dataclasses.dataclass(frozen=True)
class MetricThresholds:
    localized_position: float = 0.30        # metres
    localized_yaw_deg: float = 10.0
    hold_time: float = 2.0                  # seconds localized without a break = converged
    diverged_position: float = 1.0
    diverged_yaw_deg: float = 30.0


def pose_errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position error in m, absolute yaw error in deg) between pose arrays (..., 3)."""
    position = np.linalg.norm(estimate[..., :2] - truth[..., :2], axis=-1)
    yaw = np.degrees(np.abs(tf.angle_difference(estimate[..., 2], truth[..., 2])))
    return position, yaw


def mirror_errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Errors with respect to the half-turn mirror image of the true pose."""
    return pose_errors(estimate, mirror_pose(truth))


def first_sustained(flags: np.ndarray, hold_steps: int, start: int = 0,
                    stop: int | None = None) -> int | None:
    """First index in [start, stop) from which ``flags`` stays True for ``hold_steps`` steps.

    A run of True that reaches ``stop`` also counts, as long as it is not empty.
    """
    stop = len(flags) if stop is None else stop
    run_start = None
    for i in range(start, stop):
        if flags[i]:
            run_start = i if run_start is None else run_start
            if i - run_start + 1 >= hold_steps:
                return run_start
        else:
            run_start = None
    return run_start


def _stats(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    if len(values) == 0:
        return {f"{prefix}_rmse": None, f"{prefix}_mae": None, f"{prefix}_p95": None}
    return {f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
            f"{prefix}_mae": float(np.mean(np.abs(values))),
            f"{prefix}_p95": float(np.percentile(np.abs(values), 95))}


def tracking_mask(localized: np.ndarray, kidnap_steps, hold_steps: int
                  ) -> tuple[np.ndarray, int | None, list[int | None]]:
    """(tracking-window mask, convergence step, recovery step per kidnap)."""
    count = len(localized)
    mask = np.zeros(count, dtype=bool)
    boundaries = [0, *sorted(kidnap_steps), count]
    starts: list[int | None] = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        converged_at = first_sustained(localized, hold_steps, begin, end)
        starts.append(converged_at)
        if converged_at is not None:
            mask[converged_at:end] = True
    return mask, starts[0], starts[1:]


def compute_metrics(result, thresholds: MetricThresholds | None = None) -> dict[str, Any]:
    """Metrics of one run (a `RunResult` from evaluation/benchmark.py)."""
    th = thresholds or MetricThresholds()
    dt = result.dt
    hold = max(1, int(round(th.hold_time / dt)))
    pos, yaw = result.position_error, result.yaw_error_deg
    mirror_pos, mirror_yaw = result.mirror_position_error, result.mirror_yaw_error_deg

    localized = (pos < th.localized_position) & (yaw < th.localized_yaw_deg)
    mirror_localized = (mirror_pos < th.localized_position) & (mirror_yaw < th.localized_yaw_deg)
    either = localized | mirror_localized
    diverged = (pos > th.diverged_position) | (yaw > th.diverged_yaw_deg)
    wrong_half = mirror_localized & ~localized

    mask, converged_at, recovered_at = tracking_mask(localized, result.kidnap_steps, hold)
    _, symmetric_converged_at, symmetric_recovered_at = tracking_mask(
        either, result.kidnap_steps, hold)
    tail = slice(max(0, len(pos) - hold), None)

    def seconds(step: int | None, origin: int = 0) -> float | None:
        return None if step is None else float((step - origin) * dt)

    kidnaps = sorted(result.kidnap_steps)
    latency_ms = 1e3 * result.update_time_s
    metrics: dict[str, Any] = {
        "steps": int(len(pos)),
        "duration_s": float(len(pos) * dt),
        "converged": converged_at is not None,
        "convergence_time_s": seconds(converged_at),
        "convergence_time_symmetric_s": seconds(symmetric_converged_at),
        **_stats(pos[mask], "position"),
        **_stats(yaw[mask], "yaw"),
        **{f"{k}_all": v for k, v in _stats(pos, "position").items()},
        **{f"{k}_all": v for k, v in _stats(yaw, "yaw").items()},
        "tracking_fraction": float(mask.mean()),
        "diverged_fraction": float(diverged[mask].mean()) if mask.any() else None,
        "final_position_error": float(pos[tail].mean()),
        "final_yaw_error": float(yaw[tail].mean()),
        "final_localized": bool(localized[tail].mean() > 0.5),
        "final_diverged": bool(localized[tail].mean() <= 0.5),
        "wrong_half_fraction": float(wrong_half.mean()),
        "final_wrong_half": bool(wrong_half[tail].mean() > 0.5),
        "kidnap_times_s": [float(k * dt) for k in kidnaps],
        "recovery_times_s": [seconds(r, k) for r, k in zip(recovered_at, kidnaps)],
        "recovery_times_symmetric_s": [seconds(r, k)
                                       for r, k in zip(symmetric_recovered_at, kidnaps)],
        "filter_converged_fraction": float(result.filter_converged.mean()),
        "overconfident_fraction": float((result.filter_converged & ~localized).mean()),
        "ambiguity_flag_fraction": float(result.symmetric_ambiguity.mean()),
        "mean_confidence": float(result.confidence.mean()),
        "mean_ess_fraction": float(result.ess_fraction.mean()),
        "resample_rate": float(result.resampled.mean()),
        "mean_injected_per_update": float(result.injected.mean()),
        "mean_landmarks_per_frame": float(result.num_landmarks.mean()),
        "mean_line_points_per_frame": float(result.num_line_points.mean()),
        "update_latency_ms_mean": float(latency_ms.mean()),
        "update_latency_ms_median": float(np.median(latency_ms)),
        "update_latency_ms_p95": float(np.percentile(latency_ms, 95)),
        "update_latency_ms_max": float(latency_ms.max()),
        "wall_time_s": float(result.wall_time_s),
    }
    return metrics


def aggregate_metrics(results: list, per_run: list[dict[str, Any]],
                      thresholds: MetricThresholds | None = None) -> dict[str, Any]:
    """Combine the runs (seeds) of one scenario.

    Error statistics are pooled over the tracking windows of all runs, so a seed with a long
    tracking window weighs more than one that converged late. Rates are fractions of runs.
    """
    th = thresholds or MetricThresholds()
    pooled_pos, pooled_yaw, pooled_pos_all, pooled_yaw_all, latency = [], [], [], [], []
    for result in results:
        hold = max(1, int(round(th.hold_time / result.dt)))
        localized = ((result.position_error < th.localized_position)
                     & (result.yaw_error_deg < th.localized_yaw_deg))
        mask, _, _ = tracking_mask(localized, result.kidnap_steps, hold)
        pooled_pos.append(result.position_error[mask])
        pooled_yaw.append(result.yaw_error_deg[mask])
        pooled_pos_all.append(result.position_error)
        pooled_yaw_all.append(result.yaw_error_deg)
        latency.append(1e3 * result.update_time_s)
    latency = np.concatenate(latency)

    def mean_of(key: str) -> float | None:
        values = [m[key] for m in per_run if m[key] is not None]
        return float(np.mean(values)) if values else None

    def flat(key: str) -> list[float]:
        return [v for m in per_run for v in m[key] if v is not None]

    kidnaps = sum(len(m["kidnap_times_s"]) for m in per_run)
    recoveries, sym_recoveries = flat("recovery_times_s"), flat("recovery_times_symmetric_s")
    return {
        "runs": len(per_run),
        "convergence_rate": float(np.mean([m["converged"] for m in per_run])),
        "convergence_time_s_mean": mean_of("convergence_time_s"),
        "convergence_rate_symmetric": float(np.mean(
            [m["convergence_time_symmetric_s"] is not None for m in per_run])),
        "convergence_time_symmetric_s_mean": mean_of("convergence_time_symmetric_s"),
        **_stats(np.concatenate(pooled_pos), "position"),
        **_stats(np.concatenate(pooled_yaw), "yaw"),
        **{f"{k}_all": v for k, v in _stats(np.concatenate(pooled_pos_all), "position").items()},
        **{f"{k}_all": v for k, v in _stats(np.concatenate(pooled_yaw_all), "yaw").items()},
        "tracking_fraction": mean_of("tracking_fraction"),
        "diverged_fraction": mean_of("diverged_fraction"),
        "divergence_rate": float(np.mean([m["final_diverged"] for m in per_run])),
        "wrong_half_rate": float(np.mean([m["final_wrong_half"] for m in per_run])),
        "wrong_half_fraction": mean_of("wrong_half_fraction"),
        "kidnaps": kidnaps,
        "recovery_rate": float(len(recoveries) / kidnaps) if kidnaps else None,
        "recovery_time_s_mean": float(np.mean(recoveries)) if recoveries else None,
        "recovery_rate_symmetric": float(len(sym_recoveries) / kidnaps) if kidnaps else None,
        "recovery_time_symmetric_s_mean": float(np.mean(sym_recoveries)) if sym_recoveries else None,
        "overconfident_fraction": mean_of("overconfident_fraction"),
        "ambiguity_flag_fraction": mean_of("ambiguity_flag_fraction"),
        "mean_ess_fraction": mean_of("mean_ess_fraction"),
        "resample_rate": mean_of("resample_rate"),
        "update_latency_ms_mean": float(latency.mean()),
        "update_latency_ms_median": float(np.median(latency)),
        "update_latency_ms_p95": float(np.percentile(latency, 95)),
        "update_latency_ms_max": float(latency.max()),
        "wall_time_s_total": float(sum(m["wall_time_s"] for m in per_run)),
    }
