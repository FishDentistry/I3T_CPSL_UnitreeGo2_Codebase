string label
string requested_target
float32 score

int32 x_min
int32 y_min
int32 x_max
int32 y_max

bool has_depth
uint32 depth_sample_count
float64 depth_pixel_u
float64 depth_pixel_v

bool has_camera_position
geometry_msgs/Point camera_position

bool has_map_position
geometry_msgs/Point map_position

# Narrow horizontal band estimated from aligned depth inside the detection.
bool has_grasp_band
geometry_msgs/Point grasp_band_camera_position
bool has_grasp_band_map_position
geometry_msgs/Point grasp_band_map_position
float32 grasp_band_width_m
float32 grasp_band_height_m
float64 grasp_band_pixel_u
float64 grasp_band_pixel_v
