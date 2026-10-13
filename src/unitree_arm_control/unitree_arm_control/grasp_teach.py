"""Command-line client for recording object-relative D1 grasp examples."""

import argparse
import sys

import rclpy
from rclpy.node import Node
from unitree_arm.srv import StartGraspTeaching
from unitree_arm.srv import CaptureGraspPose
from unitree_arm.srv import SaveGraspTeaching
from unitree_arm.srv import CancelGraspTeaching


def _parser():
    parser = argparse.ArgumentParser(
        description='Record achieved D1 poses as a class-level grasp template.'
    )
    commands = parser.add_subparsers(dest='operation', required=True)
    start = commands.add_parser('start', help='freeze a detected band reference')
    start.add_argument('object_class')
    start.add_argument('--object-id', default='')
    start.add_argument('--band-index', type=int, default=0,
                       help='1-based detected band; 0 selects the first')
    capture = commands.add_parser('capture', help='record achieved gripper TF')
    capture.add_argument('stage', choices=('pregrasp', 'contact'))
    save = commands.add_parser('save', help='persist a verified demonstration')
    save.add_argument('name')
    save.add_argument('--confirmed', action='store_true',
                      help='confirm a physical grasp succeeded')
    commands.add_parser('cancel', help='discard the active teaching session')
    return parser


def main(args=None):
    options = _parser().parse_args(args)
    if options.operation == 'start' and options.band_index < 0:
        raise SystemExit('--band-index must not be negative')
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
            request.band_index = options.band_index
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
            print('object_id={} band_width_m={:.4f}'.format(
                result.selected_object_id, result.band_width_m
            ))
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
