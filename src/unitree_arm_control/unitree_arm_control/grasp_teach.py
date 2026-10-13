"""Command-line client for recording object-relative D1 grasp examples."""

import argparse
import sys

from unitree_arm_control import grasp_templates
from unitree_arm_control import grasping

# Must match the coordinator's grasp_templates_path; pass --path otherwise.
DEFAULT_LIBRARY_PATH = '~/.local/share/unitree_arm_control/grasp_templates.yaml'


def _parser():
    parser = argparse.ArgumentParser(
        description='Record achieved D1 poses as a class-level grasp template.'
    )
    commands = parser.add_subparsers(dest='operation', required=True)
    start = commands.add_parser(
        'start', help='freeze a detected object-center reference'
    )
    start.add_argument('object_class')
    start.add_argument('--object-id', default='')
    capture = commands.add_parser(
        'capture',
        help='record achieved gripper TF; contact is required, pregrasp is '
             'optional and only sets how far back the approach starts',
    )
    capture.add_argument('stage', choices=('pregrasp', 'contact'))
    save = commands.add_parser('save', help='persist a verified demonstration')
    save.add_argument('name')
    save.add_argument('--confirmed', action='store_true',
                      help='confirm a physical grasp succeeded')
    commands.add_parser('cancel', help='discard the active teaching session')
    listing = commands.add_parser(
        'list', help='show taught grasps (optionally for one class)'
    )
    listing.add_argument('object_class', nargs='?', default='')
    listing.add_argument('--path', default=DEFAULT_LIBRARY_PATH,
                         help='grasp library file (default: %(default)s)')
    remove = commands.add_parser(
        'remove', help='delete one taught grasp by class and name'
    )
    remove.add_argument('object_class')
    remove.add_argument('name')
    remove.add_argument('--path', default=DEFAULT_LIBRARY_PATH,
                        help='grasp library file (default: %(default)s)')
    return parser


def _describe(template):
    details = [
        'enabled' if template.get('enabled', True) else 'DISABLED',
        'approach from {:.1f} cm'.format(
            100.0 * grasp_templates.approach_distance(template)
        ),
        'joint angles saved' if template.get('contact_joints') is not None
        else 'no joint angles (taught before they were recorded)',
    ]
    contact = template['contact']['position']
    details.append('contact {:+.1f} {:+.1f} {:+.1f} cm from object center'.format(
        *(100.0 * value for value in contact)
    ))
    return '  {:<24} {}'.format(template['name'], ', '.join(details))


def _library_command(options):
    try:
        library = grasp_templates.GraspTemplateLibrary(options.path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('cannot read {}: {}'.format(options.path, error), file=sys.stderr)
        return 1
    if options.operation == 'list':
        wanted = grasping.normalize_label(options.object_class)
        classes = {
            label: items for label, items in sorted(library.classes.items())
            if not wanted or label == wanted
        }
        print('taught grasps in {}'.format(library.path))
        if not classes:
            print('  (none{})'.format(' for ' + wanted if wanted else ''))
            return 0
        for label, items in classes.items():
            print('{}:'.format(label))
            for template in items:
                print(_describe(template))
        return 0
    try:
        library.remove(options.object_class, options.name)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print('removed {} from {} in {}; the running coordinator reloads the '
          'file before its next request'.format(
              options.name,
              grasping.normalize_label(options.object_class),
              library.path,
          ))
    return 0


def main(args=None):
    options = _parser().parse_args(args)
    if options.operation in ('list', 'remove'):
        return _library_command(options)
    import rclpy
    from rclpy.node import Node
    from unitree_arm.srv import StartGraspTeaching
    from unitree_arm.srv import CaptureGraspPose
    from unitree_arm.srv import SaveGraspTeaching
    from unitree_arm.srv import CancelGraspTeaching
    if options.operation == 'save' and not options.confirmed:
        raise SystemExit('save requires --confirmed after a successful grasp')
    services = {
        'start': (StartGraspTeaching, '/d1_grasp/teach/start'),
        'capture': (CaptureGraspPose, '/d1_grasp/teach/capture'),
        'save': (SaveGraspTeaching, '/d1_grasp/teach/save'),
        'cancel': (CancelGraspTeaching, '/d1_grasp/teach/cancel'),
    }
    service_type, service_name = services[options.operation]
    rclpy.init()
    node = Node('d1_grasp_teach_client')
    try:
        client = node.create_client(service_type, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('{} is unavailable'.format(service_name))
        request = service_type.Request()
        if options.operation == 'start':
            request.object_class = options.object_class
            request.object_id = options.object_id
        elif options.operation == 'capture':
            request.stage = options.stage
        elif options.operation == 'save':
            request.name = options.name
            request.grasp_succeeded = options.confirmed
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        if not future.done():
            raise RuntimeError('{} timed out'.format(service_name))
        result = future.result()
        if result is None:
            raise RuntimeError('service returned no result')
        print(result.message)
        if options.operation == 'start' and result.success:
            print('object_id={}'.format(result.selected_object_id))
        if options.operation == 'capture' and result.success:
            point = result.achieved_pose.pose.position
            print('achieved {} in {}: [{:.4f}, {:.4f}, {:.4f}] m'.format(
                options.stage, result.achieved_pose.header.frame_id,
                point.x, point.y, point.z,
            ))
        if not result.success:
            return 1
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
