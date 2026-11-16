"""The dataset generator: configuration, plausible viewpoints, randomization presets,
determinism, manifests and leakage-free splits. Tests that render are marked ``render``."""
import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from robosoccer_perception.datasets import randomization, schema  # noqa: E402
from robosoccer_perception.datasets.generator import (  # noqa: E402
    DatasetConfig, ScenarioRenderer, generate_dataset, scenario_seed_key)
from robosoccer_perception.datasets.io import SoccerDataset, validate_dataset  # noqa: E402
from robosoccer_perception.datasets.sampling import DEFAULT_MODE_WEIGHTS, FrameSampler  # noqa: E402
from robosoccer_perception.geometry.camera import StereoCameraConfig  # noqa: E402

CAMERA = StereoCameraConfig.load().with_resolution(320, 180)


# ------------------------------------------------------------------ configuration
@pytest.mark.parametrize("name", ["smoke", "baseline", "large"])
def test_dataset_presets_load(name):
    cfg = DatasetConfig.load(name)
    assert set(cfg.splits) == {"train", "val", "test"}
    assert cfg.num_frames == sum(s.scenarios * s.frames_per_scenario for s in cfg.splits.values())
    assert cfg.labels.supersample >= 1 and cfg.content["max_robots"] >= 1
    assert set(cfg.modes) <= set(DEFAULT_MODE_WEIGHTS)
    assert DatasetConfig.from_dict(cfg.to_dict()) == cfg, "a recorded configuration reloads"
    assert DatasetConfig.load("smoke").num_frames < 100 < DatasetConfig.load("baseline").num_frames


def test_dataset_config_overrides_and_validation(tiny_dataset_config):
    cfg = DatasetConfig.load("smoke", {"seed": 5, "resolution": [1280, 720]})
    assert (cfg.seed, cfg.width, cfg.height) == (5, 1280, 720)
    per_split = DatasetConfig.from_dict({"name": "x", "randomization": "mild", "splits": {
        "train": {"scenarios": 1, "frames_per_scenario": 1},
        "test": {"scenarios": 1, "frames_per_scenario": 1, "randomization": "strong"}}})
    assert (per_split.splits["train"].randomization, per_split.splits["test"].randomization) \
        == ("mild", "strong")
    with pytest.raises(ValueError, match="split"):
        DatasetConfig.from_dict({"name": "x", "splits": {"holdout": {"scenarios": 1,
                                                                     "frames_per_scenario": 1}}})
    with pytest.raises(ValueError, match="image_format"):
        DatasetConfig.from_dict({"name": "x", "image_format": "gif", "splits": {}})


def test_every_scenario_has_its_own_seed():
    keys = {tuple(scenario_seed_key(3, split, index))
            for split in schema.SPLITS for index in range(50)}
    assert len(keys) == 150
    assert scenario_seed_key(3, "val", 0) != scenario_seed_key(3, "train", 0)
    assert scenario_seed_key(3, "train", 1) != scenario_seed_key(4, "train", 1)


# ------------------------------------------------------------------ randomization presets
@pytest.mark.parametrize("name", randomization.PRESETS)
def test_presets_load_and_sample_deterministically(s_field, name):
    preset = randomization.load_preset(name)
    assert preset["name"] == name
    first = randomization.sample_scene_overrides(np.random.default_rng(4), preset, s_field, 3)
    again = randomization.sample_scene_overrides(np.random.default_rng(4), preset, s_field, 3)
    other = randomization.sample_scene_overrides(np.random.default_rng(5), preset, s_field, 3)
    assert first == again and len(first["robots"]) == 3
    json.dumps(first)                                    # recordable in a manifest
    if name == "none":
        assert set(first) == {"robots"} and first == other, "nothing but the default robots"
    else:
        assert first != other
        assert {"turf", "line_rgb", "lighting", "goal", "ball", "background"} <= set(first)
        for rgb in (first["turf"]["base_rgb"], first["line_rgb"], first["goal"]["rgb"]):
            assert all(0.0 <= c <= 1.0 for c in rgb)
        assert first["turf"]["base_rgb"][1] == max(first["turf"]["base_rgb"]), "turf stays green"
        assert first["lighting"]["lights"][0]["direction"][2] < 0, "light shines downward"


def test_strong_varies_more_than_mild(s_field):
    def turf_value_spread(name):
        preset = randomization.load_preset(name)
        values = [max(randomization.sample_scene_overrides(
            np.random.default_rng(i), preset, s_field, 1)["turf"]["base_rgb"]) for i in range(60)]
        return np.std(values)

    assert turf_value_spread("strong") > 1.5 * turf_value_spread("mild")


def test_camera_mount_stays_physically_plausible():
    mount = CAMERA.mount
    rng = np.random.default_rng(0)
    none = randomization.load_preset("none")
    for _ in range(20):
        rig = randomization.sample_rig(rng, none, mount)
        assert (rig.height, rig.pitch, rig.roll) == (mount.height, mount.pitch, 0.0)
        assert abs(math.degrees(rig.pan)) <= 30.0
    for name in ("mild", "strong"):
        preset = randomization.load_preset(name)["camera"]
        rigs = [randomization.sample_rig(rng, {"camera": preset}, mount) for _ in range(400)]
        heights = np.array([r.height for r in rigs])
        pitches = np.degrees([r.pitch for r in rigs])
        rolls = np.degrees([r.roll for r in rigs])
        assert preset["height_limits"][0] <= heights.min() and heights.max() <= preset["height_limits"][1]
        assert preset["pitch_limits_deg"][0] <= pitches.min() and pitches.max() <= preset["pitch_limits_deg"][1]
        assert np.abs(rolls).max() <= preset["roll_limit_deg"] + 1e-9
        assert heights.mean() == pytest.approx(0.90, abs=0.03), "centred on the placeholder mount"
        assert pitches.mean() == pytest.approx(25.0, abs=2.5)
        assert heights.std() > 0.02 and pitches.std() > 2.0


# ------------------------------------------------------------------ frame layouts
@pytest.mark.parametrize("preset_name", ["none", "strong"])
def test_frame_sampler_modes_give_valid_deterministic_layouts(s_field, preset_name):
    preset = randomization.load_preset(preset_name)
    sampler = FrameSampler(s_field, CAMERA, preset)
    assert sampler.mode_probabilities.sum() == pytest.approx(1.0)
    for mode in DEFAULT_MODE_WEIGHTS:
        for seed in range(25):
            spec = sampler.sample(np.random.default_rng(seed), mode)
            assert spec == sampler.sample(np.random.default_rng(seed), mode)
            assert spec.mode == mode and len(spec.robots) == 3
            assert s_field.contains(spec.base_pose[:2]), "the camera stands on the carpet"
            assert -math.pi <= spec.base_pose[2] < math.pi
            low, high = preset["camera"]["pitch_limits_deg"]
            assert low - 1e-6 <= math.degrees(spec.rig.pitch) <= high + 1e-6
            assert np.linalg.norm(spec.ball_quat) == pytest.approx(1.0)
            if spec.ball_xy is not None:
                assert s_field.contains(spec.ball_xy), "the ball is on the carpet"
            json.dumps(spec.to_dict())
            if mode == "empty_turf":
                assert spec.ball_xy is None and not any(spec.robots)
            if mode.startswith("ball_"):
                assert spec.ball_xy is not None
            if mode in ("robot_on_line", "robot_close", "ball_occluded"):
                assert spec.robots[0] is not None
    with pytest.raises(ValueError):
        sampler.sample(np.random.default_rng(0), "moonwalk")
    with pytest.raises(ValueError):
        FrameSampler(s_field, CAMERA, preset, mode_weights={"moonwalk": 1.0})


def test_edge_case_modes_put_things_where_they_claim(s_field):
    preset = randomization.load_preset("mild")
    sampler = FrameSampler(s_field, CAMERA, preset)
    rng = np.random.default_rng(1)
    far = [sampler.sample(rng, "ball_far") for _ in range(40)]
    distances = [math.dist(s.base_pose[:2], s.ball_xy) for s in far]
    assert np.median(distances) > 4.0
    near_line = [sampler.sample(rng, "ball_near_line") for _ in range(40)]
    gaps = s_field.distance_to_paint(np.array([s.ball_xy for s in near_line]))
    assert gaps.max() <= 0.07 + 1e-9, "the ball touches or overlaps a line"
    on_line = [sampler.sample(rng, "robot_on_line") for _ in range(40)]
    assert s_field.distance_to_paint(np.array([s.robots[0].xy for s in on_line])).max() < 1e-6
    midfield = [sampler.sample(rng, "midfield") for _ in range(40)]
    assert max(abs(s.base_pose[0]) for s in midfield) <= 0.18 * s_field.dims.length
    touchline = [sampler.sample(rng, "touchline") for _ in range(40)]
    assert min(abs(s.base_pose[1]) for s in touchline) >= s_field.half_width - 0.5 - 1e-9
    close = [sampler.sample(rng, "robot_close") for _ in range(40)]
    assert max(math.dist(s.base_pose[:2], s.robots[0].xy) for s in close) <= 1.0 + 1e-9
    modes = [sampler.sample(rng).mode for _ in range(600)]
    assert set(modes) == set(DEFAULT_MODE_WEIGHTS), "every mode occurs in a normal draw"
    assert modes.count("random") > modes.count("empty_turf")


# ------------------------------------------------------------------ generated dataset
@pytest.mark.render
def test_generated_dataset_is_complete_and_consistent(tiny_dataset, tiny_dataset_config):
    statistics = validate_dataset(tiny_dataset)
    assert statistics["frames"] == tiny_dataset_config.num_frames == 20
    assert statistics["frames_per_split"] == {"train": 12, "val": 4, "test": 4}
    assert sum(statistics["boxes_per_class"].values()) > 10
    dataset = SoccerDataset(tiny_dataset)
    assert dataset.splits == ["train", "val", "test"] and len(dataset) == 20
    for record in dataset:
        image, mask = dataset.image(record), dataset.mask(record)
        assert image.shape == (180, 320, 3) and image.dtype == np.uint8
        assert mask.shape == (180, 320) and mask.dtype == np.uint8, "mask is pixel-aligned"
        assert set(np.unique(mask)) <= {0, 1, 2}
        assert sum(record["semantic_pixels"].values()) == 320 * 180
        assert record["semantic_pixels"]["field_line"] == int((mask == schema.FIELD_LINE).sum())
        boxes = dataset.boxes(record)
        labelled = [obj for obj in record["objects"] if obj["labeled"]]
        assert [cls for cls, _ in boxes] == [obj["class_id"] for obj in labelled]
        for (_, box), obj in zip(boxes, labelled):
            np.testing.assert_allclose(box, obj["bbox_xyxy"], atol=2e-3 * 320)
            assert -1e-3 <= box[0] < box[2] <= 320 + 1e-3 and -1e-3 <= box[1] < box[3] <= 180 + 1e-3
        assert hashlib.sha1((dataset.root / record["image"]).read_bytes()).hexdigest() \
            == record["image_sha1"]


@pytest.mark.render
def test_manifest_records_everything_needed_to_reproduce(tiny_dataset, tiny_dataset_config):
    dataset = SoccerDataset(tiny_dataset)
    header = dataset.header
    assert header["format_version"] == schema.DATASET_FORMAT_VERSION
    assert header["dataset_version"] == "test" and header["seed"] == 11
    assert DatasetConfig.from_dict(header["config"]) == tiny_dataset_config
    assert header["field"]["name"] == "s_field" and header["field"]["dimensions"]["length"] == 9.0
    assert header["camera"]["width"] == 320 and header["camera"]["fx"] == pytest.approx(160.0)
    assert header["camera"]["approximate"] is True and header["camera"]["mount"]["height"] == 0.9
    assert header["detection_classes"] == list(schema.DETECTION_CLASSES)
    assert header["semantic_classes"] == list(schema.SEMANTIC_CLASSES)
    assert header["label_policy"] == tiny_dataset_config.labels.to_dict()
    assert header["randomization_presets"]["mild"] == randomization.load_preset("mild")
    assert header["counts"]["train"]["frames"] == 12
    record = dataset.frames("train")[0]
    for key in ("frame_id", "split", "scenario_id", "image", "label", "mask", "randomization",
                "frame", "photometric", "objects", "T_field_left_optical", "image_sha1"):
        assert key in record
    assert {"mode", "base_pose_xy_yaw", "rig", "ball_xy", "robots"} <= set(record["frame"])
    assert {"height", "pitch", "roll", "pan"} <= set(record["frame"]["rig"])
    assert set(record["photometric"]) and record["randomization"] == "mild"
    assert len(record["objects"]) == 1 + 4 + 3, "ball, four posts, three robots: labelled or not"
    scenarios = dataset.scenarios()
    assert len(scenarios) == 5 and all("scene_overrides" in s and "seed_key" in s for s in scenarios)
    yolo = (dataset.root / "data.yaml").read_text()
    assert "images/train" in yolo and "goal_post" in yolo and "path:" not in yolo


@pytest.mark.render
def test_generation_is_deterministic(tiny_dataset, tiny_dataset_config, tmp_path):
    again = generate_dataset(tiny_dataset_config, tmp_path / "again")
    first, second = SoccerDataset(tiny_dataset), SoccerDataset(again)
    assert first.header["manifest_sha256"] == second.header["manifest_sha256"]
    for a, b in zip(first.records, second.records):
        assert a == b
        for key in ("image", "label", "mask"):
            assert (first.root / a[key]).read_bytes() == (second.root / b[key]).read_bytes()
    other_seed = generate_dataset(
        dataclasses.replace(tiny_dataset_config, seed=12,
                            splits={"train": tiny_dataset_config.splits["val"]}),
        tmp_path / "other")
    assert SoccerDataset(other_seed).records[0]["image_sha1"] != first.records[0]["image_sha1"]


@pytest.mark.render
def test_splits_share_no_scenario_appearance_or_viewpoint(tiny_dataset):
    dataset = SoccerDataset(tiny_dataset)
    by_split = {split: {r["scenario_id"] for r in dataset.frames(split)} for split in dataset.splits}
    assert not (by_split["train"] & by_split["val"]) and not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])
    scenarios = dataset.scenarios()
    assert len({tuple(s["seed_key"]) for s in scenarios}) == len(scenarios)
    turf = [tuple(s["scene_overrides"]["turf"]["base_rgb"]) for s in scenarios]
    seeds = [s["scene_overrides"]["turf"]["seed"] for s in scenarios]
    assert len(set(turf)) == len(turf) and len(set(seeds)) == len(seeds), "no shared appearance"
    poses = [tuple(np.round(r["frame"]["base_pose_xy_yaw"], 6)) for r in dataset]
    assert len(set(poses)) == len(poses), "no viewpoint appears twice, within or across splits"
    assert len({r["image_sha1"] for r in dataset}) == len(dataset)


@pytest.mark.render
def test_a_scenario_does_not_depend_on_the_rest_of_the_dataset(gl_context, tiny_dataset,
                                                               tiny_dataset_config):
    """Growing or reordering a dataset leaves existing scenarios untouched."""
    bigger = dataclasses.replace(tiny_dataset_config, splits={
        "test": dataclasses.replace(tiny_dataset_config.splits["test"], scenarios=9),
        "train": dataclasses.replace(tiny_dataset_config.splits["train"], scenarios=50)})
    dataset = SoccerDataset(tiny_dataset)
    with ScenarioRenderer(bigger, "train", 2) as scenario:
        frame = scenario.render(3)
        record = next(r for r in dataset if r["frame_id"] == "train_0002_003")
        assert frame.spec.to_dict() == record["frame"]
        np.testing.assert_array_equal(frame.labels.semantic, dataset.mask(record))
        assert scenario.record()["scene_overrides"] == next(
            s for s in dataset.scenarios() if s["scenario_id"] == "train_0002")["scene_overrides"]


@pytest.mark.render
def test_existing_directories_are_not_overwritten_by_accident(tiny_dataset, tiny_dataset_config,
                                                              tmp_path):
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(tiny_dataset_config, tiny_dataset)
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.txt").write_text("do not delete")
    with pytest.raises(FileExistsError, match="not a generated dataset"):
        generate_dataset(tiny_dataset_config, precious, overwrite=True)
    assert (precious / "thesis.txt").read_text() == "do not delete"
    with pytest.raises(FileNotFoundError):
        SoccerDataset(precious)


@pytest.mark.render
def test_validation_catches_broken_labels(tiny_dataset_config, tmp_path):
    one = dataclasses.replace(tiny_dataset_config, splits={
        "train": dataclasses.replace(tiny_dataset_config.splits["train"], scenarios=1)})
    root = generate_dataset(one, tmp_path / "broken")
    validate_dataset(root)
    dataset = SoccerDataset(root)
    record = next(r for r in dataset if any(o["labeled"] for o in r["objects"]))
    (root / record["label"]).write_text("7 0.5 0.5 0.1 0.1\n" * sum(o["labeled"] for o in record["objects"]))
    with pytest.raises(ValueError, match="malformed|class"):
        validate_dataset(root)


@pytest.mark.render
@pytest.mark.parametrize("preset", randomization.PRESETS)
def test_every_preset_renders_labelled_frames(gl_context, tiny_dataset_config, preset):
    cfg = dataclasses.replace(tiny_dataset_config, splits={
        "train": dataclasses.replace(tiny_dataset_config.splits["train"], randomization=preset)})
    with ScenarioRenderer(cfg, "train", 0) as scenario:
        frames = [scenario.render(i) for i in range(3)]
    for frame in frames:
        assert frame.image.shape == (180, 320, 3) and frame.image.std() > 5
        assert frame.labels.semantic.shape == (180, 320)
        assert bool(frame.photometric) == (preset != "none")
    if preset == "none":
        assert set(scenario.scene_overrides) == {"robots"}
