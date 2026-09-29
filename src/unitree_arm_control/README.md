# Unitree D1 arm control

`unitree_arm_control` is a ROS 2 Foxy-compatible wrapper around the D1 arm's
existing `/arm_Command` and `/arm_Feedback` topics. It does not use or modify
Unitree SDK2. The node publishes the same `unitree_arm/msg/ArmString` JSON that
the arm already accepts through CycloneDDS.

The payload layout follows Unitree's public D1 examples, including the
`seq`/`address`/`funcode` envelope and `angle0` through `angle6` fields:
<https://github.com/chen37058/Grasp-with-the-Unitree-D1/tree/main/src>

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

# Move to the measured lay-down pose, confirm it from feedback, then release
# all joints. Support the arm and keep its entire path clear before calling.
ros2 service call /d1_arm_controller/lay_down_and_release \
  unitree_arm/srv/LayDownArm "{}"

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

# Watch receipt and execution acknowledgements.
ros2 topic echo /d1_arm_controller/command_result
```

Topic names are launch arguments, so a suffixed D1 topic can be selected
without changing the arm firmware or this package:

```bash
ros2 launch unitree_arm_control d1_arm.launch.py \
  feedback_topic:=/arm_Feedback_1 command_topic:=/arm_Command_1
```
