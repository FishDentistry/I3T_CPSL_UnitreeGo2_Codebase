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


def _contact(reference, point, approach):
    return grasp_templates.relative_pose(
        reference, point, grasping.quaternion_from_approach(approach)
    )


def _template_with(pregrasp, contact):
    return {
        'name': 'derived',
        'reference': 'target_object_center',
        'pregrasp': pregrasp,
        'contact': contact,
        'gripper_open_degrees': 49.0,
        'enabled': True,
    }


def test_contact_only_teaching_derives_straight_pregrasp():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    contact = _contact(reference, (0.50, 0.0, 0.10), (1.0, 0.0, 0.0))
    pregrasp, standoff = grasp_templates.straight_pregrasp(contact, 0.10)
    assert standoff == pytest.approx(0.10)
    assert pregrasp['position'] == pytest.approx([-0.10, 0.0, 0.0])
    grasp_templates.validate_template(_template_with(pregrasp, contact))


def test_sloppy_demonstrated_pregrasp_is_straightened():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    contact = _contact(reference, (0.50, 0.0, 0.10), (1.0, 0.0, 0.0))
    # 6 cm sideways, 4 cm high and a twisted wrist: rejected before.
    sloppy = grasp_templates.relative_pose(
        reference, (0.38, 0.06, 0.14),
        grasping.quaternion_from_approach((1.0, 0.3, -0.2)),
    )
    with pytest.raises(ValueError):
        grasp_templates.validate_template(_template_with(sloppy, contact))
    pregrasp, standoff = grasp_templates.straight_pregrasp(
        contact, 0.10, sloppy
    )
    assert standoff == pytest.approx(0.12)
    assert pregrasp['position'] == pytest.approx([-0.12, 0.0, 0.0])
    assert pregrasp['orientation'] == pytest.approx(contact['orientation'])
    grasp_templates.validate_template(_template_with(pregrasp, contact))


def test_angled_and_top_down_contacts_are_accepted():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    for approach in ((1.0, 0.0, -1.0), (0.0, 0.0, -1.0), (1.0, 0.5, -0.3)):
        contact = _contact(reference, (0.50, 0.0, 0.12), approach)
        pregrasp, _ = grasp_templates.straight_pregrasp(contact, 0.10)
        grasp_templates.validate_template(_template_with(pregrasp, contact))


def test_approach_back_toward_arm_is_rejected():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    contact = _contact(reference, (0.50, 0.0, 0.10), (-1.0, 0.0, 0.0))
    pregrasp, _ = grasp_templates.straight_pregrasp(contact, 0.10)
    with pytest.raises(ValueError, match='back toward the arm'):
        grasp_templates.validate_template(_template_with(pregrasp, contact))


def test_pregrasp_in_front_of_contact_is_rejected():
    reference = grasp_templates.object_reference(
        (0.50, 0.0, 0.10), (0.0, 0.0, 0.0)
    )
    contact = _contact(reference, (0.50, 0.0, 0.10), (1.0, 0.0, 0.0))
    ahead = _contact(reference, (0.51, 0.0, 0.10), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='behind the contact'):
        grasp_templates.straight_pregrasp(contact, 0.10, ahead)


def test_yaw_offsets_try_taught_side_first_and_cover_range():
    import math
    offsets = grasp_templates.yaw_offsets(math.radians(30), math.radians(15))
    assert offsets == pytest.approx(
        [0.0] + [math.radians(v) for v in (15, -15, 30, -30)]
    )
    full = grasp_templates.yaw_offsets(math.pi, math.radians(15))
    assert len(full) == 24
    assert len({round(math.degrees(v)) % 360 for v in full}) == 24
    assert grasp_templates.yaw_offsets(0.0, 0.1) == (0.0,)


def test_rotated_reference_moves_grasp_around_object_center():
    import math
    template = _demonstration()
    center = (0.50, 0.0, 0.10)
    taught = grasp_templates.candidate_from_template(
        template, grasp_templates.object_reference(center, (0.0, 0.0, 0.0))
    )
    turned = grasp_templates.candidate_from_template(
        template,
        grasp_templates.object_reference(
            center, (0.0, 0.0, 0.0), math.pi / 2.0
        ),
        math.pi / 2.0,
    )
    # Pre-grasp was 15 cm on the arm side; a quarter turn puts it at -Y.
    assert taught['pregrasp_point'] == pytest.approx((0.35, 0.0, 0.10))
    assert turned['pregrasp_point'] == pytest.approx((0.50, -0.15, 0.10))
    assert turned['grasp_point'] == pytest.approx(taught['grasp_point'])
    assert turned['yaw_offset'] == pytest.approx(math.pi / 2.0)
    assert grasping.approach_axis_error(
        turned['orientation'], turned['approach_direction']
    ) == pytest.approx(0.0)


def test_class_yaw_ranges_parse_and_reject_bad_values():
    import math
    ranges = grasp_templates.parse_class_yaw_ranges(
        ['Cup=3.1416', 'soda can=1.0']
    )
    assert ranges['cup'] == pytest.approx(math.pi)
    assert ranges['soda can'] == pytest.approx(1.0)
    for bad in (['cup'], ['cup=-1'], ['cup=4'], ['cup=1', 'CUP=2']):
        with pytest.raises(ValueError):
            grasp_templates.parse_class_yaw_ranges(bad)


def test_optional_contact_joints_are_validated():
    template = _demonstration()
    template['contact_joints'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    grasp_templates.validate_template(template)
    template['contact_joints'] = [0.1, 0.2]
    with pytest.raises(ValueError, match='contact_joints'):
        grasp_templates.validate_template(template)
