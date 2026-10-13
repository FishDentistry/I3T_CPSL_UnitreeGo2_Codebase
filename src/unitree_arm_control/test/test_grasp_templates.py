"""Geometry and persistence checks for taught grasp templates."""

import pytest

from unitree_arm_control import grasp_templates
from unitree_arm_control import grasping


def _demonstration():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    orientation = grasping.quaternion_from_approach((1.0, 0.0, 0.0))
    return {
        'name': 'side',
        'reference': 'target_object_center',
        'pregrasp': grasp_templates.relative_pose(
            reference, (0.35, 0.0, 0.10), orientation
        ),
        'contact': grasp_templates.relative_pose(
            reference, (0.50, 0.0, 0.10), orientation
        ),
        'gripper_open_degrees': 49.0,
        'enabled': True,
    }


def test_relocates_demonstration_with_object_center_azimuth():
    template = _demonstration()
    grasp_templates.validate_template(template)
    moved = grasp_templates.object_reference(
        (0.0, 0.50, 0.10), (0.0, 0.0, 0.0)
    )
    candidate = grasp_templates.candidate_from_template(
        template, moved
    )
    assert candidate['pregrasp_point'] == pytest.approx((0.0, 0.35, 0.10))
    assert candidate['grasp_point'] == pytest.approx((0.0, 0.50, 0.10))
    assert grasping.approach_axis_error(
        candidate['orientation'], candidate['approach_direction']
    ) == pytest.approx(0.0)


def test_candidate_is_anchored_to_object_center_without_band_metadata():
    candidate = grasp_templates.candidate_from_template(
        _demonstration(),
        grasp_templates.object_reference(
            (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
        ),
    )
    assert candidate['pregrasp_point'][0] == pytest.approx(0.35)
    assert candidate['grasp_point'][0] == pytest.approx(0.50)
    assert candidate['band_width'] is None
    assert candidate['band_target_point'] == pytest.approx((0.50, 0.0, 0.10))


def test_band_referenced_template_is_rejected():
    template = _demonstration()
    template['reference'] = 'selected_grasp_band'
    with pytest.raises(ValueError, match='target_object_center'):
        grasp_templates.validate_template(template)


def test_library_round_trip_and_duplicate_rejection(tmp_path):
    path = tmp_path / 'grasp_templates.yaml'
    library = grasp_templates.GraspTemplateLibrary(str(path))
    template = _demonstration()
    library.add('Cup', template)
    loaded = grasp_templates.GraspTemplateLibrary(str(path))
    assert loaded.templates_for(' cup ')[0]['name'] == 'side'
    with pytest.raises(ValueError, match='already exists'):
        loaded.add('cup', template)


def test_version_one_library_requires_reteaching(tmp_path):
    path = tmp_path / 'grasp_templates.yaml'
    path.write_text('version: 1\nclasses: {}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='obsolete version 1'):
        grasp_templates.GraspTemplateLibrary(str(path))
