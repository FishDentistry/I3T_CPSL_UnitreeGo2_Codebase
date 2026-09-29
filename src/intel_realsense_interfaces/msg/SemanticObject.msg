uint8 STATUS_ACTIVE=0
uint8 STATUS_STALE=1
uint8 STATUS_REMOVED=2

string object_id
string label
geometry_msgs/Point position
float32 confidence
uint32 observation_count
builtin_interfaces/Time first_observed
builtin_interfaces/Time last_observed
uint32 movement_count
uint8 status
