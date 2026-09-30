uint8 STAGE_REJECTED=0
uint8 STAGE_ACCEPTED=1
uint8 STAGE_PLANNING=2
uint8 STAGE_PLAN_READY=3
uint8 STAGE_EXECUTING=4
uint8 STAGE_PREGRASP_REACHED=5
uint8 STAGE_FAILED=6

std_msgs/Header header
string request_id
string object_class
string object_id
uint8 stage
bool terminal
bool success
string message
geometry_msgs/PoseStamped target_pose
