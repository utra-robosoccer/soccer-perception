"""Image pixel -> ground point: rays, the ground intersection, rejection of unusable rays, the
effect of a wrong camera estimate, and uncertainty propagation. Pure NumPy except for one
rendering test that ties the projection to what MuJoCo actually draws."""
import math

import numpy as np
import pytest

from robosoccer_perception.geometry import ground_projection as gp
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import (
    CameraIntrinsics, RigState, StereoCameraConfig, StereoCameraModel)
from robosoccer_perception.observations.noise import apply_pitch_error


@pytest.fixture(scope="module")
def camera_cfg() -> StereoCameraConfig:
    return StereoCameraConfig.load("zed_mini_sim").with_resolution(640, 360)


@pytest.fixture(scope="module")
def calibration(camera_cfg) -> gp.EstimatedCameraCalibration:
    return gp.EstimatedCameraCalibration.from_camera_config(camera_cfg)


def _exact(calibration, **changes):
    """The same calibration with every uncertainty switched off except what is named."""
    zero = dict(pixel_std=0.0, height_std=0.0, pitch_std_deg=0.0, roll_std_deg=0.0,
                yaw_std_deg=0.0, focal_rel_std=0.0)
    return calibration.replace(uncertainty=gp.CalibrationUncertainty(**{**zero, **changes}))


# ------------------------------------------------------------------ frames and conventions
@pytest.mark.parametrize("rig", [
    dict(), dict(pan=0.4), dict(pitch=math.radians(40.0)), dict(roll=math.radians(5.0)),
    dict(pan=-0.7, pitch=math.radians(12.0), roll=math.radians(-3.0), height=1.05)])
def test_estimated_pose_uses_the_same_frame_chain_as_the_phase_1_camera_model(camera_cfg,
                                                                              calibration, rig):
    """One chain, two users: what is rendered and what is back-projected must agree exactly."""
    state = camera_cfg.mount.replace(**rig)
    expected = StereoCameraModel(camera_cfg).T_base_camera("left", state)
    pose = calibration.pose(pan=state.pan, pitch=state.pitch, roll=state.roll, height=state.height)
    np.testing.assert_allclose(pose.T_base_camera, expected, atol=1e-12)


def test_left_and_right_cameras_differ_by_the_baseline(camera_cfg):
    left = gp.EstimatedCameraCalibration.from_camera_config(camera_cfg, "left").pose()
    right = gp.EstimatedCameraCalibration.from_camera_config(camera_cfg, "right").pose()
    offset = left.T_base_camera[:3, 3] - right.T_base_camera[:3, 3]
    np.testing.assert_allclose(offset, [0.0, camera_cfg.baseline, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        gp.EstimatedCameraCalibration.from_camera_config(camera_cfg, "middle")


def test_pixel_rays_follow_the_optical_convention(calibration):
    k = calibration.intrinsics
    centre, right, below = gp.pixel_rays([[k.cx, k.cy], [k.cx + 50, k.cy], [k.cx, k.cy + 50]], k)
    np.testing.assert_allclose(centre, [0.0, 0.0, 1.0], atol=1e-12)
    assert right[0] > 0 and abs(right[1]) < 1e-12, "u grows to the right = optical +x"
    assert below[1] > 0 and abs(below[0]) < 1e-12, "v grows downward = optical +y"
    np.testing.assert_allclose(np.linalg.norm([centre, right, below], axis=1), 1.0)


def test_the_optical_axis_meets_the_ground_where_trigonometry_says(calibration):
    k = calibration.intrinsics
    pose = calibration.pose()
    assert pose.height == pytest.approx(0.90) and math.degrees(pose.depression) == pytest.approx(25.0)
    ground = gp.project_to_ground([[k.cx, k.cy]], calibration, pose)
    # Camera 0.9 m up, 5 cm ahead of the base, looking 25 deg down; the left camera sits half
    # a baseline to the left.
    assert ground.valid[0]
    np.testing.assert_allclose(ground.points[0],
                               [0.05 + 0.90 / math.tan(math.radians(25.0)), 0.0315], atol=1e-9)
    assert ground.depression[0] == pytest.approx(math.radians(25.0))


def test_projection_inverts_the_camera_model(calibration, rng):
    """Random ground points -> pixels (forward model) -> ground (this module): identity."""
    for _ in range(20):
        pose = calibration.pose(pan=rng.uniform(-1.0, 1.0), pitch=rng.uniform(0.25, 0.8),
                                roll=rng.uniform(-0.1, 0.1), height=rng.uniform(0.6, 1.2))
        points = np.column_stack([rng.uniform(0.5, 5.0, 40), rng.uniform(-3.0, 3.0, 40)])
        pixels, seen = gp.ground_to_pixels(points, calibration, pose)
        assert seen.any()
        back = gp.project_to_ground(pixels[seen], calibration, pose,
                                    gp.ProjectionLimits(0.5, 50.0, 0.0, 0.0))
        assert back.valid.all()
        np.testing.assert_allclose(back.points, points[seen], atol=1e-9)
        np.testing.assert_allclose(back.range, np.linalg.norm(points[seen], axis=1), atol=1e-9)


def test_projection_is_vectorized_and_shapes_are_stable(calibration):
    pose = calibration.pose()
    empty = gp.project_to_ground(np.zeros((0, 2)), calibration, pose)
    assert len(empty) == 0 and empty.valid.shape == (0,)
    many = gp.project_to_ground(np.tile([[320.0, 300.0]], (5000, 1)), calibration, pose)
    assert many.valid.all() and np.ptp(many.points, axis=0).max() == 0.0
    assert gp.projection_covariance(np.zeros((0, 2)), calibration, pose).total.shape == (0, 2, 2)


# ------------------------------------------------------------------ rejection
def test_rays_that_cannot_be_trusted_are_flagged_with_the_reason(calibration):
    k = calibration.intrinsics
    pose = calibration.pose()
    limits = gp.ProjectionLimits(min_depression_deg=4.0, max_range=6.0, min_range=0.5,
                                 image_margin_px=2.0)
    horizon_row = k.cy - k.fy * math.tan(math.radians(25.0))        # where depression is zero

    def row_for(depression_deg: float) -> float:
        return k.cy + k.fy * math.tan(math.radians(depression_deg - 25.0))

    pixels = np.array([
        [k.cx, horizon_row - 20.0],     # 0 above the horizon
        [k.cx, row_for(2.0)],           # 1 just below it: 2 deg of depression
        [k.cx, row_for(6.0)],           # 2 8.6 m away
        [k.cx, row_for(20.0)],          # 3 fine: 2.5 m
        [k.cx, k.height - 1.0],         # 4 inside the border margin
        [-30.0, k.cy],                  # 5 outside the image
        [np.nan, 10.0],                 # 6 not a number
    ])
    ground = gp.project_to_ground(pixels, calibration, pose, limits)
    S = gp.ProjectionStatus
    assert list(ground.valid) == [False, False, False, True, False, False, False]
    assert ground.status[0] & S.ABOVE_HORIZON and np.isnan(ground.points[0]).all()
    assert ground.status[1] & S.NEAR_HORIZON
    assert ground.status[2] == S.TOO_FAR and ground.range[2] == pytest.approx(8.6, abs=0.1)
    assert ground.status[3] == S.VALID and ground.range[3] == pytest.approx(2.52, abs=0.02)
    assert ground.status[4] & S.OUTSIDE_IMAGE
    assert ground.status[5] & S.OUTSIDE_IMAGE
    assert ground.status[6] & S.UNSTABLE
    assert ground.count(S.TOO_FAR) == 2, "the near-horizon ray (26 m) is also too far"
    assert ground.count(S.ABOVE_HORIZON) >= 1
    close = gp.project_to_ground([[k.cx, row_for(70.0)]], calibration, pose, limits)
    assert close.status[0] & S.TOO_CLOSE, "0.33 m ahead: nearer than min_range"


def test_a_camera_looking_up_sees_no_ground(calibration):
    pose = calibration.pose(pitch=math.radians(-40.0))
    k = calibration.intrinsics
    v, u = np.mgrid[0:k.height:20, 0:k.width:40]
    ground = gp.project_to_ground(np.column_stack([u.ravel(), v.ravel()]), calibration, pose)
    assert not ground.valid.any()
    assert (ground.status & gp.ProjectionStatus.ABOVE_HORIZON).all()


def test_a_camera_below_the_ground_is_rejected(calibration):
    pose = calibration.pose(height=-0.1)
    ground = gp.project_to_ground([[320.0, 300.0]], calibration, pose)
    assert not ground.valid[0] and ground.status[0] & gp.ProjectionStatus.UNSTABLE


# ------------------------------------------------------------------ wrong camera estimates
@pytest.mark.parametrize("error_deg", [-2.0, -0.5, 0.5, 2.0])
def test_a_pitch_error_moves_ranges_the_way_phase_1_modelled_it(calibration, error_deg):
    """The oracle's `apply_pitch_error` and real back-projection must tell the same story."""
    truth = calibration.pose()
    believed = truth.perturbed(d_pitch=math.radians(error_deg))
    camera_xy = truth.T_base_camera[:2, 3]
    true_range = np.array([1.0, 2.0, 3.0, 5.0])
    points = camera_xy + np.column_stack([true_range, np.zeros(4)])
    pixels, _ = gp.ground_to_pixels(points, calibration, truth)          # what the camera sees
    wrong = gp.project_to_ground(pixels, calibration, believed,
                                 gp.ProjectionLimits(0.1, 100.0, 0.0, 0.0))
    measured = np.linalg.norm(wrong.points - camera_xy, axis=1)
    # Believing the camera looks further down than it does puts every point CLOSER.
    assert np.all((measured < true_range) == (error_deg > 0))
    np.testing.assert_allclose(
        measured, apply_pitch_error(true_range, truth.height, math.radians(error_deg)), rtol=1e-6)
    assert abs(measured[3] - 5.0) > 4.0 * abs(measured[0] - 1.0), "far points move much more"


def test_a_height_error_scales_every_range(calibration):
    truth = calibration.pose()
    camera_xy = truth.T_base_camera[:2, 3]
    points = camera_xy + np.array([[1.0, 0.3], [2.5, -1.0], [4.0, 2.0]])
    pixels, _ = gp.ground_to_pixels(points, calibration, truth)
    for error in (-0.10, 0.05):
        wrong = gp.project_to_ground(pixels, calibration, truth.perturbed(d_height=error))
        scale = (truth.height + error) / truth.height
        np.testing.assert_allclose(wrong.points - camera_xy, scale * (points - camera_xy),
                                   atol=1e-9)


def test_yaw_and_roll_errors_act_in_the_expected_directions(calibration):
    truth = calibration.pose()
    points = np.array([[3.0, 0.0], [3.0, 1.0]])
    pixels, _ = gp.ground_to_pixels(points, calibration, truth)
    yaw = math.radians(3.0)
    turned = gp.project_to_ground(pixels, calibration, truth.perturbed(d_yaw=yaw)).points
    expected = points @ tf.rot_z(yaw)[:2, :2].T
    np.testing.assert_allclose(turned, expected, atol=1e-9)
    # Positive roll tips the camera's right side down: a point seen on the right is then
    # believed to be nearer, one on the left farther.
    side = np.array([[3.0, -1.5], [3.0, 1.5]])
    pixels, _ = gp.ground_to_pixels(side, calibration, truth)
    rolled = gp.project_to_ground(pixels, calibration,
                                  truth.perturbed(d_roll=math.radians(2.0))).points
    assert np.linalg.norm(rolled[0]) < np.linalg.norm(side[0])
    assert np.linalg.norm(rolled[1]) > np.linalg.norm(side[1])


def test_perturbation_composes_like_the_mount_parameters(calibration):
    base = calibration.pose(pan=0.3)
    np.testing.assert_allclose(
        base.perturbed(d_height=0.07, d_pitch=0.05).T_base_camera,
        calibration.pose(pan=0.3, pitch=calibration.nominal_pitch + 0.05,
                         height=calibration.nominal_height + 0.07).T_base_camera, atol=1e-12)
    # Roll turns the camera about its own optical centre; in the mount chain it turns the rig
    # about the rig centre, half a baseline away. Same orientation, sub-millimetre shift.
    rolled = base.perturbed(d_roll=0.02).T_base_camera
    mounted = calibration.pose(pan=0.3, roll=0.02).T_base_camera
    np.testing.assert_allclose(rolled[:3, :3], mounted[:3, :3], atol=1e-12)
    assert np.abs(rolled[:3, 3] - mounted[:3, 3]).max() < 1e-3
    assert base.perturbed().T_base_camera == pytest.approx(base.T_base_camera)
    assert math.degrees(base.heading) == pytest.approx(math.degrees(0.3), abs=1e-6)


def test_a_wrong_focal_length_or_principal_point_shifts_the_ground_point(calibration):
    truth = calibration.pose()
    points = np.array([[3.0, 1.0]])
    pixels, _ = gp.ground_to_pixels(points, calibration, truth)
    longer = gp.project_to_ground(pixels, calibration.with_intrinsics_error(focal_scale=1.05), truth)
    shifted = gp.project_to_ground(pixels, calibration.with_intrinsics_error(cx_shift=10.0), truth)
    assert abs(longer.points[0, 1]) < 1.0, "a longer focal length narrows the believed view"
    assert shifted.points[0, 1] > 1.0, "principal point believed further right: point further left"
    assert calibration.with_intrinsics_error().intrinsics == calibration.intrinsics


def test_imu_tilt_is_applied_about_the_base(calibration):
    level = calibration.pose()
    tilted = level.with_base_tilt(roll=0.0, pitch=math.radians(5.0))
    assert math.degrees(tilted.depression) == pytest.approx(30.0, abs=1e-9)
    assert tilted.height < level.height, "pitching the torso forward lowers a forward camera"
    np.testing.assert_allclose(level.with_base_tilt(0.0, 0.0).T_base_camera, level.T_base_camera)


# ------------------------------------------------------------------ uncertainty
def test_pitch_uncertainty_matches_the_closed_form(calibration):
    pose = calibration.pose()
    camera_x = pose.T_base_camera[0, 3]
    lateral = pose.T_base_camera[1, 3]
    ranges = np.array([1.5, 3.0, 5.0])
    points = np.column_stack([camera_x + ranges, np.full(3, lateral)])
    pixels, _ = gp.ground_to_pixels(points, calibration, pose)
    covariance = gp.projection_covariance(pixels, _exact(calibration, pitch_std_deg=1.0), pose)
    along = covariance.std_along([1.0, 0.0])
    expected = gp.range_sensitivity_to_pitch(ranges, pose.height) * math.radians(1.0)
    np.testing.assert_allclose(along, expected, rtol=1e-4)
    np.testing.assert_allclose(covariance.std_along([0.0, 1.0]), 0.0, atol=1e-6)
    assert np.all(covariance.pixel == 0.0)


def test_pixel_uncertainty_grows_with_range_and_is_larger_along_the_view(calibration):
    pose = calibration.pose()
    points = np.array([[1.5, 0.0], [3.0, 0.0], [5.0, 0.0]])
    pixels, _ = gp.ground_to_pixels(points, calibration, pose)
    covariance = gp.projection_covariance(pixels, _exact(calibration, pixel_std=1.0), pose)
    range_std, bearing_std = covariance.polar(points)
    assert np.all(np.diff(range_std) > 0)
    assert np.all(range_std > points[:, 0] * bearing_std), "one pixel is longer than it is wide"
    # About (r^2 + h^2) / (f h) metres per pixel along the view: 9 cm at 5 m for fx = 320.
    assert range_std[2] == pytest.approx((25.0 + 0.81) / (320.0 * 0.9), rel=0.1)
    assert np.all(covariance.camera == 0.0)
    doubled = gp.projection_covariance(pixels, _exact(calibration, pixel_std=2.0), pose)
    np.testing.assert_allclose(doubled.pixel, 4.0 * covariance.pixel, rtol=1e-9)
    override = gp.projection_covariance(pixels, _exact(calibration), pose, pixel_std=2.0)
    np.testing.assert_allclose(override.pixel, doubled.pixel, rtol=1e-9)


def test_covariance_parts_are_separate_and_add_up(calibration):
    pose = calibration.pose(pan=0.5)
    points = np.array([[2.0, 1.0], [4.0, 2.5]])
    pixels, _ = gp.ground_to_pixels(points, calibration, pose)
    covariance = gp.projection_covariance(pixels, calibration, pose)
    np.testing.assert_allclose(covariance.total, covariance.pixel + covariance.camera)
    for part in (covariance.pixel, covariance.camera):
        assert np.allclose(part, part.transpose(0, 2, 1)), "symmetric"
        assert np.all(np.linalg.eigvalsh(part) > -1e-12), "positive semi-definite"
    assert np.all(covariance.std_rms("camera") > covariance.std_rms("pixel")), \
        "with the default beliefs the camera pose dominates beyond a couple of metres"
    # Anisotropic pixel noise: an uncertain row hurts the range, an uncertain column the bearing.
    rows = gp.projection_covariance(pixels, _exact(calibration), pose, pixel_std=(0.0, 2.0))
    columns = gp.projection_covariance(pixels, _exact(calibration), pose, pixel_std=(2.0, 0.0))
    assert rows.polar(points)[0][0] > columns.polar(points)[0][0]
    assert columns.polar(points)[1][0] > rows.polar(points)[1][0]


def test_each_camera_parameter_contributes_its_own_uncertainty(calibration):
    pose = calibration.pose()
    points = np.array([[3.0, 1.5]])
    pixels, _ = gp.ground_to_pixels(points, calibration, pose)
    sizes = {}
    for name in ("height_std", "pitch_std_deg", "roll_std_deg", "yaw_std_deg", "focal_rel_std"):
        value = 0.02 if name in ("height_std", "focal_rel_std") else 1.0
        covariance = gp.projection_covariance(pixels, _exact(calibration, **{name: value}), pose)
        sizes[name] = float(covariance.std_rms("camera")[0])
        assert sizes[name] > 1e-3, name
    assert sizes["pitch_std_deg"] > max(sizes["roll_std_deg"], sizes["yaw_std_deg"]), \
        "pitch is the parameter that matters most for ground projection"
    # A yaw error swings the point sideways by range * angle.
    yaw = gp.projection_covariance(pixels, _exact(calibration, yaw_std_deg=1.0), pose)
    assert yaw.polar(points)[1][0] == pytest.approx(math.radians(1.0), rel=1e-3)


def test_uncertainty_near_the_horizon_is_huge_not_undefined(calibration):
    k = calibration.intrinsics
    pose = calibration.pose()
    horizon_row = k.cy - k.fy * math.tan(math.radians(25.0))
    covariance = gp.projection_covariance([[k.cx, horizon_row + 0.2], [k.cx, horizon_row - 5.0]],
                                          calibration, pose)
    assert np.isfinite(covariance.total).all()
    assert np.all(covariance.std_rms() > 50.0)


# ------------------------------------------------------------------ what counts as an estimate
def test_only_the_robots_own_estimate_is_accepted(calibration):
    pose = calibration.pose()
    rig = RigState(0.9, math.radians(25.0))
    for wrong_pose in (pose.T_base_camera, rig, None, "pose"):
        with pytest.raises(TypeError):
            gp.project_to_ground([[320.0, 300.0]], calibration, wrong_pose)
    for wrong_calibration in (calibration.intrinsics, StereoCameraConfig.load("zed_mini_sim"), None):
        with pytest.raises(TypeError):
            gp.project_to_ground([[320.0, 300.0]], wrong_calibration, pose)
        with pytest.raises(TypeError):
            gp.projection_covariance([[320.0, 300.0]], wrong_calibration, pose)
        with pytest.raises(TypeError):
            gp.ground_to_pixels([[2.0, 0.0]], wrong_calibration, pose)


def test_an_estimated_pose_must_be_a_rigid_transform():
    with pytest.raises(ValueError):
        gp.EstimatedCameraPose(np.eye(3))
    skewed = np.eye(4)
    skewed[0, 1] = 0.3
    with pytest.raises(ValueError):
        gp.EstimatedCameraPose(skewed)
    with pytest.raises(ValueError):
        gp.CalibrationUncertainty(pitch_std_deg=-1.0)


def test_calibration_follows_the_image_resolution():
    full = StereoCameraConfig.load("zed_mini_sim")
    half = full.with_resolution(640, 360)
    a = gp.EstimatedCameraCalibration.from_camera_config(full)
    b = gp.EstimatedCameraCalibration.from_camera_config(half)
    point = np.array([[3.0, 0.8]])
    pixel_a, _ = gp.ground_to_pixels(point, a, a.pose())
    pixel_b, _ = gp.ground_to_pixels(point, b, b.pose())
    np.testing.assert_allclose((pixel_a + 0.5) / 2.0 - 0.5, pixel_b, atol=1e-9)


# ------------------------------------------------------------------ against the renderer
@pytest.mark.render
def test_back_projection_agrees_with_rendered_images_including_an_off_centre_principal_point(
        gl_context, s_field, s_landmarks):
    """Pixels that MuJoCo paints white, back-projected with the camera model, land on paint.

    The intrinsics are deliberately asymmetric (fx != fy, principal point off centre): if this
    module disagreed with the renderer about the principal point's sign (MuJoCo's horizontal
    convention is the opposite of OpenCV's; see `_add_pinhole_camera`), the points would miss.
    """
    import dataclasses

    from robosoccer_perception.simulation.mujoco_scene import LEFT_CAMERA, MujocoScene

    base = StereoCameraConfig.load("zed_mini_sim")
    skewed = dataclasses.replace(base, intrinsics=CameraIntrinsics(640, 360, 300.0, 330.0,
                                                                   280.0, 200.0))
    calibration = gp.EstimatedCameraCalibration.from_camera_config(skewed)
    pose_field = np.array([-2.6, -1.4, math.radians(35.0)])
    with MujocoScene(s_field, skewed) as scene:
        scene.teleport(pose_field)
        scene.set_ball(None)
        image = scene.render(LEFT_CAMERA, show_platform=False).astype(int)
    white = (image.min(axis=2) > 170) & (np.ptp(image, axis=2) < 25)
    v, u = np.nonzero(white)
    ground = gp.project_to_ground(np.column_stack([u, v]), calibration, calibration.pose(),
                                  gp.ProjectionLimits(4.0, 4.0, 0.3, 3.0))
    assert ground.valid.sum() > 300
    in_field = tf.se2_transform_points(pose_field, ground.points[ground.valid])
    distance = s_field.distance_to_paint(in_field)
    assert np.mean(distance < 0.03) > 0.97, "white pixels back-project onto paint"
    # The same pixels with the principal point mirrored (the sign mistake) do not.
    mirrored = calibration.replace(intrinsics=CameraIntrinsics(640, 360, 300.0, 330.0,
                                                               639.0 - 280.0, 200.0))
    wrong = gp.project_to_ground(np.column_stack([u, v]), mirrored, mirrored.pose(),
                                 gp.ProjectionLimits(4.0, 4.0, 0.3, 3.0))
    off = s_field.distance_to_paint(tf.se2_transform_points(pose_field, wrong.points[wrong.valid]))
    assert np.mean(off < 0.03) < 0.5
