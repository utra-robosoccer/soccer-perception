"""Every scenario the viewer can run: the Phase 3 scenarios (a rigid camera on the placeholder
platform) and the Phase 4 scenarios (a moving body, the humanoid model, an IMU)."""
from __future__ import annotations

from typing import Any

from robosoccer_perception.humanoid_benchmark import scenarios as humanoid_scenarios
from robosoccer_perception.humanoid_benchmark.scenarios import HumanoidScenario
from robosoccer_perception.viewer.humanoid_session import HumanoidLiveSession, SessionOptions
from robosoccer_perception.viewer.session import LiveSession
from robosoccer_perception.vision_benchmark import scenarios as vision_scenarios


def all_scenarios() -> list[Any]:
    """Phase 4 scenarios first (they are what the viewer opens with), then Phase 3's."""
    return [*humanoid_scenarios.SCENARIOS.values(), *vision_scenarios.SCENARIOS.values()]


def find_scenario(text: str) -> Any:
    """The scenario with this id, or the first whose id starts with ``text``."""
    scenarios = all_scenarios()
    for scenario in scenarios:
        if scenario.id == text:
            return scenario
    matches = [s for s in scenarios if s.id.startswith(text)]
    if not matches:
        raise KeyError(f"no scenario matches '{text}' "
                       f"(available: {[s.id for s in scenarios]})")
    return matches[0]


def is_phase4(scenario: Any) -> bool:
    return isinstance(scenario, HumanoidScenario)


def open_session(scenario: Any, seed: int, resources, *, options: SessionOptions | None = None,
                 max_particles: int = 600, with_truth: bool = True,
                 max_steps: int | None = None):
    """A live session for either kind of scenario. ``options`` applies to Phase 4 only."""
    if is_phase4(scenario):
        return HumanoidLiveSession(scenario, seed, resources, options=options,
                                   max_particles=max_particles, with_truth=with_truth,
                                   max_steps=max_steps)
    return LiveSession(scenario, seed, resources, max_particles=max_particles,
                       with_truth=with_truth, max_steps=max_steps)
