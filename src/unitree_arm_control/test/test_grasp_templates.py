"""Geometry and persistence checks for taught grasp templates."""

import pytest

from unitree_arm_control import grasp_templates
from unitree_arm_control import grasping


def _demonstration():
    reference = grasp_templates.band_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    orientation = grasping.quaternion_from_approach((1.0, 0.0, 0.0))
    return {
        'name': 'side',
        'reference': 'selected_grasp_band',
        'taught_band_width_m': 0.05,
        'pregrasp': grasp_templates.relative_pose(
            reference, (0.35, 0.0, 0.10), orientation
        ),
        'contact': grasp_templates.relative_pose(
            reference, (0.50, 0.0, 0.10), orientation
        ),
        'gripper_open_degrees': 49.0,
        'enabled': True,
    }


def test_relocates_demonstration_with_band_azimuth():
    template = _demonstration()
    grasp_templates.validate_template(template)
    moved = grasp_templates.band_reference(
        (0.0, 0.50, 0.10), (0.0, 0.0, 0.0)
    )
    candidate = grasp_templates.candidate_from_template(
        template, moved, 0.05, 0.75, 1.25
    )
    assert candidate['pregrasp_point'] == pytest.approx((0.0, 0.35, 0.10))
    assert candidate['grasp_point'] == pytest.approx((0.0, 0.50, 0.10))
    assert grasping.approach_axis_error(
        candidate['orientation'], candidate['approach_direction']
    ) == pytest.approx(0.0)


def test_wider_band_increases_pregrasp_clearance():
    candidate = grasp_templates.candidate_from_template(
        _demonstration(),
        grasp_templates.band_reference(
            (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
        ),
        0.06, 0.75, 1.25,
    )
    assert candidate['pregrasp_point'][0] == pytest.approx(0.345)
    assert candidate['grasp_point'][0] == pytest.approx(0.50)


def test_out_of_range_width_keeps_template_out_of_candidate_list():
    assert grasp_templates.candidate_from_template(
        _demonstration(),
        grasp_templates.band_reference(
            (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
        ),
        0.07, 0.75, 1.25,
    ) is None


def test_library_round_trip_and_duplicate_rejection(tmp_path):
    path = tmp_path / 'grasp_templates.yaml'
    library = grasp_templates.GraspTemplateLibrary(str(path))
    template = _demonstration()
    library.add('Cup', template)
    loaded = grasp_templates.GraspTemplateLibrary(str(path))
    assert loaded.templates_for(' cup ')[0]['name'] == 'side'
    with pytest.raises(ValueError, match='already exists'):
        loaded.add('cup', template)
