std_msgs/Header header

# Unique caller-provided identifier used to correlate status messages.
string request_id

# Required semantic class, for example "cup".
string object_class

# Optional stable semantic-map ID. When empty, the closest eligible object of
# object_class is selected.
string object_id
