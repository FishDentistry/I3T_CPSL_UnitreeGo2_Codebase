# Unitree D1 arm control

`unitree_arm_control` is a ROS 2 Foxy-compatible wrapper around the D1 arm's
existing `/arm_Command` and `/arm_Feedback` topics. It does not use or modify
Unitree SDK2. The node publishes the same `unitree_arm/msg/ArmString` JSON that
the arm already accepts through CycloneDDS.

The payload layout follows Unitree's public D1 examples, including the
`seq`/`address`/`funcode` envelope and `angle0` through `angle6` fields:
<https://github.com/chen37058/Grasp-with-the-Unitree-D1/tree/main/src>

## Joint states and TF

Each valid joint-angle feedback message is also published as a
`sensor_msgs/msg/JointState` on `/joint_states`. The message contains only
the D1 names (`d1_joint_0` through `d1_joint_5` and `d1_gripper_joint`), so it
can share the topic with another publisher that supplies the robot's leg
joints. `robot_state_publisher` combines these partial updates to publish the
arm link transforms.

J0 through J5 are converted from degrees to radians. The D1 documentation
specifies the gripper stroke but not the J6 angle-to-stroke calibration, so
J6 uses an adjustable linear mapping. The defaults map 0 to 45 degrees onto
0 to 0.03 metres of per-finger travel and clamp values outside that range.
To use measured endpoints instead:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py \
  gripper_closed_degrees:=0.0 \
  gripper_open_degrees:=45.0 \
  gripper_max_travel_m:=0.03
```

Joint states are published whenever the wrapper receives feedback, including
when command publishing is disabled or the arm motors are released. This
keeps RViz synchronized during passive monitoring and drag teaching.
When started through `dog.launch.py`, the wrapper—and therefore the D1 joint
state publisher—is present only when `launch_arm:=true`.

## Safety model

The node starts in monitor-only mode. It will parse feedback, but every command
service rejects requests until `commanding_enabled` is explicitly set to true.
By default it also refuses to publish if `/arm_Feedback` has not been received
recently. No command is published automatically at startup.

Joints 0 through 5 are checked against the published D1-550 mechanical angle
limits. Joint 6 (the gripper) is checked only for a finite value because D1
variants do not expose a consistent command range/unit for it.

## Build and verify on the robot workspace

```bash
sudo apt install ros-foxy-control-msgs
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
colcon build --packages-up-to unitree_arm_control --symlink-install
source install/setup.bash
colcon test --packages-select unitree_arm_control
colcon test-result --verbose

ros2 interface show unitree_arm/msg/ArmString
ros2 topic info /arm_Feedback --verbose
ros2 topic echo /arm_Feedback unitree_arm/msg/ArmString --once
```

Start in passive mode first:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py
ros2 topic echo /d1_arm_controller/joint_angles --once
ros2 topic echo /d1_arm_controller/status --once
```

Only after feedback is confirmed and the physical area is safe, restart with
commands enabled:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py commanding_enabled:=true
```

Examples (these can move or release the physical arm):

```bash
# Enable arm
ros2 service call /d1_arm_controller/set_arm_enabled \
  unitree_arm/srv/SetArmEnabled "{enabled: true}"

# Enable commands for the arm after launch
ros2 param set /d1_arm_controller commanding_enabled true

# Return to zero.
ros2 service call /d1_arm_controller/zero \
  unitree_arm/srv/ZeroArm "{}"

# Set joint 5 to 60 degrees.
ros2 service call /d1_arm_controller/set_joint \
  unitree_arm/srv/SetJoint \
  "{joint_id: 5, angle_degrees: 60.0}"

# Release all joints for drag teaching.
ros2 service call /d1_arm_controller/set_arm_enabled \
  unitree_arm/srv/SetArmEnabled "{enabled: false}"

# Move to the measured lay-down stowed pose, confirm it from feedback, then release
# all joints. Support the arm and keep its entire path clear before calling.
ros2 service call /d1_arm_controller/lay_down_and_release \
  unitree_arm/srv/LayDownArm "{}"
```

The measured lay-down joint values in degrees, J0 through J6, are:

```text
[-83.80000305175781, -90.0, 89.80000305175781,
 -10.600000381469727, 6.400000095367432, -2.0999999046325684,
 -0.8999999761581421]
```

The lay-down service is asynchronous: its response confirms that the motion
command was published, not that the arm has already stopped or been released.
The controller requires a successful motion execution acknowledgement and
three consecutive `/arm_Feedback` samples in which every joint is within 2
degrees of the measured pose. Only then does it publish the all-joint release
command. If the motion is rejected or these conditions are not met within 15
seconds, it does not release the joints. Completion and failure are reported in
the controller log, and command acknowledgements remain available on
`/d1_arm_controller/command_result`. Releasing the joints is not the same as
turning motor power off; this service does not send a power-off command.

The confirmation behavior can be adjusted at launch if testing shows that the
defaults need tuning:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py \
  commanding_enabled:=true \
  lay_down_tolerance_degrees:=2.0 \
  lay_down_required_samples:=3 \
  lay_down_timeout_sec:=15.0
```

```bash
# Watch receipt and execution acknowledgements.
ros2 topic echo /d1_arm_controller/command_result
```

Topic names are launch arguments, so a suffixed D1 topic can be selected
without changing the arm firmware or this package:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py \
  feedback_topic:=/arm_Feedback_1 command_topic:=/arm_Command_1
```

## Trajectory interface

The controller provides the standard
`control_msgs/action/FollowJointTrajectory` action at
`/d1_arm_controller/follow_joint_trajectory`. Trajectories must include all
six arm joints. The gripper joint is optional; when omitted, its most recent
feedback position is preserved.

Trajectory positions use URDF/ROS units:

- `d1_joint_0` through `d1_joint_5`: radians
- `d1_gripper_joint`: metres of per-finger travel

The controller linearly interpolates position waypoints and publishes D1
smooth-mode commands at 10 Hz by default. It monitors live joint feedback and
does not report success until the final position is within tolerance. A goal
is rejected when commanding is disabled, feedback is stale, a required arm
joint is missing, waypoint times are invalid, or a segment exceeds the D1
position or velocity limits. Services that command the arm are rejected while
a trajectory is active.

The command rate and final feedback checks are configurable at launch:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py \
  commanding_enabled:=true \
  trajectory_command_rate_hz:=10.0 \
  trajectory_goal_tolerance_radians:=0.01 \
  trajectory_gripper_tolerance_m:=0.005 \
  trajectory_goal_timeout_sec:=3.0
```

When starting the complete stack through `dog.launch.py`, the corresponding
launch argument is `arm_trajectory_goal_tolerance_radians`.

This action is the controller boundary intended for later MoveIt integration.
It is currently a position-only trajectory executor; supplied waypoint
velocities and accelerations are not used. Path tolerances and
velocity/acceleration goal tolerances are rejected rather than silently
ignored.

## Initial trajectory testing

Build and source the affected packages:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
colcon build --packages-up-to unitree_arm_control --symlink-install
source install/setup.bash
colcon test --packages-select unitree_arm_control
colcon test-result --verbose
```

Start the normal robot launch or arm launch:

```bash
ros2 launch go2_launcher dog.launch.py \
  launch_arm:=true arm_command_enabled:=true
```

In another sourced terminal, verify that current feedback and the trajectory
action are available before enabling commands:

```bash
ros2 topic echo /d1_arm_controller/joint_angles --once
ros2 action info /d1_arm_controller/follow_joint_trajectory
```

If the arm is not already enabled and holding its position, permit commands
and enable it:

```bash
ros2 param set /d1_arm_controller commanding_enabled true
ros2 service call /d1_arm_controller/set_arm_enabled \
  unitree_arm/srv/SetArmEnabled "{enabled: true}"
```

The following test reads the current joint state, rotates only `d1_joint_5`
(the final wrist-roll joint) by five degrees over two seconds, holds for one
second, and returns to the exact starting joint state over two seconds. This
is intentionally subtle: the expected visible motion is a small twist of the
entire gripper assembly, not opening or closing the gripper fingers. The test
automatically reverses the movement when the positive direction would cross
that joint's limit. The gripper and all other arm joints retain their measured
starting positions.

Run the test only after visually confirming that the RViz arm pose agrees with
the physical arm:

```bash
ros2 run unitree_arm_control d1_trajectory_test
```

The test checks action feedback throughout the motion and fails if it does not
observe at least half of the requested outward displacement. This prevents a
stationary arm from passing merely because the trajectory finishes at the
same position from which it started.

The test displacement can be reduced or another revolute joint selected:

```bash
ros2 run unitree_arm_control d1_trajectory_test --ros-args \
  -p joint_name:=d1_joint_5 \
  -p delta_degrees:=3.0 \
  -p move_duration_sec:=3.0 \
  -p hold_duration_sec:=1.0
```

After the default wrist-roll test succeeds, a slightly more visible wrist-pitch
test can be run with the arm's local clearance checked first:

```bash
ros2 run unitree_arm_control d1_trajectory_test --ros-args \
  -p joint_name:=d1_joint_4 \
  -p delta_degrees:=10.0 \
  -p move_duration_sec:=3.0
```

Pressing `Ctrl+C` asks the action server to cancel the active trajectory and
command the most recent feedback position as a hold target. Cancellation is a
software stop and does not replace the robot's physical emergency-stop
procedure. After the test, confirm that the action succeeded and that joint
feedback returned to the starting values:

```bash
ros2 topic echo /d1_arm_controller/joint_angles --once
```

## MoveIt 2

MoveIt configuration is contained in this package. It defines the
six-joint `d1_arm` planning group from `d1_base_link` to the
`d1_gripper_center` tool frame at the midpoint of the two fingertip faces,
KDL inverse kinematics, conservative motion limits, OMPL planning,
self-collision exclusions for adjacent arm links, and a controller mapping to
the existing
`/d1_arm_controller/follow_joint_trajectory` action. 

SDK angles J0 through J5 map directly to the ROS joint positions in radians.
The URDF joint axes encode the hardware-positive directions, including the
corrected negative axis for `d1_joint_3`; no additional sign conversion is
performed by the controller.

Install the ROS 2 Foxy MoveIt binary packages on the deployment system:

```bash
source /opt/ros/foxy/setup.bash
sudo apt update
sudo apt install ros-foxy-moveit python3-yaml
```

No MoveIt source build, Python virtual environment, MoveIt Setup Assistant
run, or Unitree SDK2 change is required for this initial configuration. The
same dependencies can subsequently be checked from the workspace with:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

Build and source the package after installing the dependencies:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
colcon build --packages-up-to unitree_arm_control --symlink-install
source install/setup.bash
```

### Planning-only verification

Starting `dog.launch.py` with `launch_arm:=true` now starts the D1 wrapper and
MoveIt `move_group`. MoveIt trajectory execution is available, but no RViz
process is started. For the first verification, keep physical arm commands
disabled so the D1 action server will reject any accidental execution request:

```bash
ros2 launch go2_launcher dog.launch.py \
  launch_arm:=true arm_command_enabled:=false
```

To inspect planning interactively, start only the supplied MoveIt RViz view in
another sourced terminal. Do not start a second `move_group` process:

```bash
ros2 launch unitree_arm_control d1_moveit.launch.py \
  start_move_group:=false
```

RViz opens with the `d1_arm` planning group selected. Confirm that the orange
interactive goal marker appears at the gripper, drag it only a small distance,
and use **Plan**. Do not use **Plan & Execute** during this stage. A successful
plan should animate in RViz without publishing an arm command.

The MoveIt planning frame is `base_link`. Targets originating in `map` must be
transformed into the robot model through TF before planning; MoveIt does not
modify the Go2 navigation transforms.

### Physical execution

When `launch_arm:=true`, `dog.launch.py` starts MoveIt with trajectory
execution enabled and without RViz. Physical motion still requires
`arm_command_enabled:=true`; the D1 wrapper remains the final hardware-command
safety gate. Before enabling it, confirm current arm feedback, model alignment,
physical clearance, and access to the hardware emergency stop.

Start the robot stack with D1 commands enabled:

```bash
ros2 launch go2_launcher dog.launch.py \
  launch_arm:=true arm_command_enabled:=true
```

If an interactive view is required, start the RViz-only launch in another
sourced terminal:

```bash
ros2 launch unitree_arm_control d1_moveit.launch.py \
  start_move_group:=false
```

Use **Plan** first and inspect the entire animated path. Only then use
**Execute**. The supplied RViz configuration starts with velocity and
acceleration scaling at 10 percent. MoveIt sends the resulting six-joint
trajectory directly to the existing D1 action server, which independently
checks feedback age, joint limits, segment velocities, cancellation, and final
position convergence.

The gripper is represented in the semantic model and collision geometry, but
is not yet exposed to MoveIt through a `GripperCommand` action. Continue using
the arm-control services for gripper commands until that action interface is
added.

## Semantic grasp coordinator

`d1_grasp_coordinator` turns a semantic-map object request into a guarded
grasp-and-release sequence. It subscribes to stable identities on
`/semantic_map`, accepts `unitree_arm/msg/GraspCommand` on
`/d1_grasp/command`, and uses fresh results from
`/grounding_dino/detection_array` before making contact.

A command contains a required semantic class and an optional stable object ID.
When the ID is empty, the closest eligible active object of that class is
selected. Before planning, the object must be active, have at least three
observations, meet the confidence threshold, have been observed within the
last 20 seconds, and satisfy the height and 0.67-metre D1 reach safeguards.

The coordinator performs the following sequence when execution is enabled:

1. Open the gripper and confirm its feedback position.
2. Generate three-dimensional pre-grasp points from the current gripper toward
   the object, including configured yaw fallbacks ordered by smallest yaw
   change.
3. Give OMPL a position-only constraint for the preferred pre-grasp point so
   that it can choose a collision-free tool orientation. If planning fails,
   repeat with the next candidate.
4. Execute the first successful MoveIt trajectory to a pre-grasp point 0.11
   metres from the configured final grasp point.
5. Require a new matching Grounding DINO observation after pre-grasp. The
   observation must remain close to the mapped point, and the correction is
   limited to 0.08 metres by default. The generated orientation points the
   tool toward the object while keeping the jaw-opening axis approximately
   horizontal.
6. Shift the detected surface point farther along the camera viewing ray by
   the configured class depth, then compute and execute a collision-checked
   MoveIt trajectory. The final target uses a 0.005-metre position region and
   a configurable orientation tolerance. The pre-grasp remains position-only
   so the long motion retains planning flexibility.
7. Close the gripper. Position feedback is accepted when it reaches the closed
   target or stalls after meaningful closure, then the object is held for
   three seconds.
8. Open the gripper and confirm release, then plan and execute a position-only
   retreat to the corrected pre-grasp point.

No lift is performed. Failures after contact cause a best-effort gripper-open
command. Only one request is processed at a time, and recent request IDs cannot
be reused.

The generated poses are published on `/d1_grasp/pregrasp_pose` and
`/d1_grasp/grasp_pose`. Progress and terminal results are published on
`/d1_grasp/status`, including opening, reacquiring, approaching, closing,
holding, releasing, retreating, and released stages.

The defaults are in `config/grasping.yaml`. `approach_yaw_offsets_rad`,
`approach_distance_m`, `position_tolerance_m`,
`grasp_position_tolerance_m`, `grasp_center_offset_m`, and the configured
gripper open/closed values are the main calibration parameters. The
grasp-center offset defaults to zero and represents measured tool geometry.

`default_forward_grasp_depth_offset_m` moves the final target beyond the
depth-derived visible surface along the viewing ray from `camera_frame`.
This correction is independent of the arm's selected approach direction, so
the full configured distance represents additional camera depth. Classes
listed in `class_forward_grasp_depth_offsets_m` override that value. Entries
use `class=metres` strings so the mapping remains compatible with ROS 2 Foxy:

```yaml
default_forward_grasp_depth_offset_m: 0.05
class_forward_grasp_depth_offsets_m:
  - "mug=0.04"
  - "bottle=0.03"
```

Class matching is case-insensitive and ignores repeated whitespace. Unlisted
classes use the default value. These offsets are a temporary approximation of
object depth until segmented three-dimensional geometry is available.

`grasp_orientation_tolerance_rad` controls the final MoveIt orientation
constraint. Its default is `0.35` radians. The generated tool Z axis follows
the selected approach direction and the gripper jaw-opening axis remains
approximately horizontal. The gripper opens to 45 degrees by default; the
controller and grasp coordinator use the same endpoint calibration.

Grounding DINO currently provides a class, bounding box, and one depth-derived
3D point rather than an object mesh or grasp pose. Consequently, the node does
not infer cup handles, object width, support surfaces, or neighboring-object
geometry. It also has gripper position feedback but no contact-force sensor.
Physical testing therefore requires a clear workspace and conservative
objects until segmented geometry and grasp-pose estimation are added.

### Plan-only command

Start the robot and MoveIt normally. The coordinator starts automatically in
plan-only mode:

```bash
ros2 launch go2_launcher dog.launch.py \
  launch_arm:=true arm_command_enabled:=false
```

The semantic-mapping stack must also be running and publishing confirmed
objects. Inspect the available identities and coordinator output:

```bash
ros2 topic echo /semantic_map
ros2 topic echo /d1_grasp/status
```

Request a specific mapped cup:

```bash
ros2 topic pub --once /d1_grasp/command \
  unitree_arm/msg/GraspCommand \
  "{request_id: 'cup_request_001', object_class: 'cup', \
  object_id: 'object_000001'}"
```

To select the closest eligible cup instead, leave `object_id` empty:

```bash
ros2 topic pub --once /d1_grasp/command \
  unitree_arm/msg/GraspCommand \
  "{request_id: 'cup_request_002', object_class: 'cup', object_id: ''}"
```

The proposed pre-grasp is published as `geometry_msgs/msg/PoseStamped` on
`/d1_grasp/pregrasp_pose`. A successful plan-only request terminates with
`STAGE_PLAN_READY`; it does not open the gripper or approach the object.

### Guarded grasp-and-release execution

After inspecting the generated pre-grasp and planned path, enable both hardware
commands and the semantic grasp sequence:

```bash
ros2 launch go2_launcher dog.launch.py \
  launch_arm:=true \
  arm_command_enabled:=true \
  grasp_execution_enabled:=true
```

Publishing a new request ID runs the complete sequence described above. A
successful request terminates with `STAGE_RELEASED`. The arm closes around the
object, holds it for `hold_duration_sec`, releases it, and retreats; it never
lifts the object.

MoveIt trajectories use D1 firmware trajectory mode (`mode: 1`) by default,
which produces substantially smoother physical motion than the legacy 10 Hz
smoothing mode. For comparison only, launch with
`arm_trajectory_command_mode:=0`.
