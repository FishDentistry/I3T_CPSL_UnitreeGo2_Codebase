cmake_minimum_required(VERSION 3.5)
project(unitree_arm)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmString.msg"
  "msg/ArmStatus.msg"
  "msg/CommandResult.msg"
  "msg/JointAngles.msg"
  "msg/MotorStatus.msg"
  "srv/LayDownArm.srv"
  "srv/SetArmEnabled.srv"
  "srv/SetJoint.srv"
  "srv/SetJointAngles.srv"
  "srv/SetJointEnabled.srv"
  "srv/SetPower.srv"
  "srv/ZeroArm.srv"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
