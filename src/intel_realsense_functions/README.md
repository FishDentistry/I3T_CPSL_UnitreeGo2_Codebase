# Intel RealSense functions

`intel_realsense_functions` provides ROS 2 nodes for publishing aligned Intel
RealSense RGB/depth data and for locating text-specified objects with Grounding
DINO.

## Nodes

### `getCameraFrames`

`getCameraFrames` connects directly to the RealSense device and publishes
aligned 640 x 480 RGB and metric depth images at 15 frames per second.
Each aligned RGB/depth pair receives the same ROS acquisition timestamp. The
message frame IDs remain empty so existing consumers can continue selecting
their camera TF frame through configuration.

| Topic | Type | Description |
| --- | --- | --- |
| `/realsense_rgb_image` | `sensor_msgs/msg/Image` | RGB image using `rgb8` encoding. |
| `/realsense_depth_image` | `sensor_msgs/msg/Image` | Aligned depth in metres using `32FC1` encoding. |
| `/depth_camera_intrinsics` | `sensor_msgs/msg/CameraInfo` | Intrinsics for the aligned color image. |

Run the node with:

```bash
ros2 run intel_realsense_functions getCameraFrames
```

### `groundingDinoNode`

`groundingDinoNode` listens for text detection targets, applies Grounding DINO
to the most recent RGB frame, estimates a robust depth from the center of each
detection, and deprojects the result into the RealSense optical coordinate
system. It then converts the point to the robot `camera_link` convention. When
the TF tree contains a transform from `camera_link` to `map`, the point is also
transformed into map coordinates.

For each depth-supported detection, the node also estimates a narrow
horizontal grasp band from the aligned depth silhouette inside the detection
box. Candidate bands exclude the top and bottom box margins, require depth
support across their height, and must be corroborated by a neighboring band.
Among bands within `grasp_band_width_tolerance_m` of the narrowest measured
width, the band closest to the box's vertical center is selected. This is a
visible-surface estimate, not a segmentation mask or a complete 3-D model;
handles, occlusions, similar-depth backgrounds, and camera angle can change
the measured width. When support is insufficient, no band is published.

The node runs inference only while at least one detection target is configured.
Detections are published continuously at the configured inference rate. It does
not publish robot or arm commands.

Inference runs in a dedicated worker so camera subscriptions remain responsive.
The node retains only one pending RGB/depth pair: frames received while the
model is busy replace the pending pair, and the next inference uses the newest
valid pair instead of processing an accumulated frame queue.

#### Topics

| Direction | Topic | Type | Description |
| --- | --- | --- | --- |
| Input | `/detection_targets` | `std_msgs/msg/String` | Requested object names. |
| Input | `/realsense_rgb_image` | `sensor_msgs/msg/Image` | RGB image. |
| Input | `/realsense_depth_image` | `sensor_msgs/msg/Image` | Aligned metric depth image. |
| Input | `/depth_camera_intrinsics` | `sensor_msgs/msg/CameraInfo` | Aligned image intrinsics. |
| Output | `/grounding_dino/detections` | `std_msgs/msg/String` | JSON detection results. |
| Output | `/grounding_dino/detection_array` | `intel_realsense_interfaces/msg/GroundedDetectionArray` | Structured detection results. |
| Output | `/grounding_dino/annotated_image` | `sensor_msgs/msg/Image` | RGB image with detection boxes. |

Detection targets may be provided as a single phrase, a comma-separated list,
a JSON list, or an object containing a `targets` list:

```bash
ros2 topic pub --once /detection_targets std_msgs/msg/String \
  "{data: 'cup'}"

ros2 topic pub --once /detection_targets std_msgs/msg/String \
  "{data: 'cup, red bottle'}"

ros2 topic pub --once /detection_targets std_msgs/msg/String \
  "{data: '[\"cup\", \"red bottle\"]'}"
```

Publishing an empty string clears the active targets and stops inference.

#### Detection result schemas

Both detection topics describe the same processed RGB/depth pair and use its
acquisition timestamp. An empty `detections` array is published when none of
the requested targets is found. Coordinates use metres.

`/grounding_dino/detection_array` is the preferred interface for ROS nodes. It
uses `intel_realsense_interfaces/msg/GroundedDetectionArray`, with one
`GroundedDetection` for each result. Boolean fields explicitly state whether
valid depth, camera coordinates, and map coordinates are present. Its header
frame ID is intentionally empty; the coordinate frames are named by the
`camera_frame` and `map_frame` fields. Optional `grasp_band_*` fields report
the estimated band center in both frames when available, its visible width
and height in metres, and its center pixel. `has_grasp_band` and
`has_grasp_band_map_position` distinguish unavailable geometry from a point
at the coordinate origin.

`/grounding_dino/detections` retains the existing JSON representation for
compatibility with current consumers:

```json
{
  "stamp": {"sec": 0, "nanosec": 0},
  "requested_targets": ["cup"],
  "camera_frame": "camera_link",
  "map_frame": "map",
  "map_transform_available": true,
  "detections": [
    {
      "label": "cup",
      "requested_target": "cup",
      "score": 0.83,
      "bounding_box_pixels": {
        "x_min": 210,
        "y_min": 120,
        "x_max": 350,
        "y_max": 410
      },
      "camera_coordinates_m": {"x": 1.2, "y": -0.1, "z": 0.0},
      "map_coordinates_m": {"x": 2.4, "y": -0.7, "z": 0.8},
      "depth_sample_count": 1530,
      "depth_pixel": {"u": 280.0, "v": 265.0},
      "grasp_band": {
        "camera_coordinates_m": {"x": 1.2, "y": -0.1, "z": 0.02},
        "map_coordinates_m": {"x": 2.4, "y": -0.7, "z": 0.82},
        "width_m": 0.05,
        "height_m": 0.015,
        "pixel_u": 280.0,
        "pixel_v": 250.0
      }
    }
  ]
}
```

`camera_coordinates_m` and `map_coordinates_m` are `null` when valid aligned
depth is unavailable. `map_coordinates_m` is also `null` when the TF lookup
fails. A missing map transform never prevents publication of pixel and camera
coordinates.
`grasp_band` is `null` when the depth profile is unreliable.

The band estimator uses NumPy already required by this package; it adds no
Python or ROS installation dependency. Its parameters are
`grasp_band_height_m` (default `0.015`),
`grasp_band_width_tolerance_m` (default `0.01`), and
`grasp_band_depth_tolerance_m` (default `0.06`). The annotated image marks
the selected band center in green with its estimated width in millimetres.
Changing `GroundedDetection.msg` requires rebuilding
`intel_realsense_interfaces` and downstream packages before launching them.
From the sourced ROS 2 Foxy workspace:

```bash
colcon build --packages-up-to intel_realsense_functions semantic_mapping unitree_arm_control --symlink-install
source install/setup.bash
```

Depth deprojection initially follows the RealSense optical convention: positive
X points right, positive Y points down, and positive Z points forward. Before
publication and TF transformation, the detector converts each point to
`camera_link` as `[x, y, z] = [z_optical, -x_optical, -y_optical]`. Published
camera coordinates therefore use positive X forward, positive Y left, and
positive Z up.

## Grounding DINO installation for Python 3.8

Grounding DINO and PyTorch are intentionally not installed by `rosdep` or by
this package's `setup.py`. They must be installed in a Python 3.8
environment appropriate for the Go2 external board and its CUDA platform.

Install the declared ROS dependencies and Python virtual-environment support:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
rosdep install --from-paths \
  src/intel_realsense_functions src/intel_realsense_interfaces \
  --ignore-src --rosdistro foxy -r -y
```

If you create an environment it must include the ROS 2 Foxy system packages:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
mkdir -p ~/.venvs
python3 -m venv --system-site-packages \
  ~/.venvs/go2-groundingdino-py38
source ~/.venvs/go2-groundingdino-py38/bin/activate
python3 --version
python3 -m pip install --upgrade "pip==24.3.1" wheel ninja
```

The displayed Python version must be 3.8.x.

Install a matching PyTorch and torchvision pair before Grounding DINO. Do not
replace a working Jetson build with the generic packages from PyPI: those wheels
are not compatible with the Jetson's ARM64 CUDA platform.

For the repository's JetPack 5.1.2 deployment target, the setup uses the
following known-compatible Python 3.8 wheels. If these versions are already
installed and the verification command below succeeds, do not uninstall or
reinstall them:

```bash
python3 -m pip install \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
python3 -m pip install --no-deps \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl
```

`--no-deps` on the torchvision installation prevents pip from replacing the
Jetson-specific PyTorch wheel with a generic build. Standard x86-64 CPU-only
development systems may instead use:

```bash
python3 -m pip install \
  "torch==2.1.2" "torchvision==0.16.2" \
  --index-url https://download.pytorch.org/whl/cpu
```

Install Python 3.8-compatible supporting packages:

```bash
python3 -m pip install \
  "numpy<2" "Pillow<11" "transformers==4.30.2" \
  "timm==0.9.12" "addict==2.4.0" "yapf==0.40.1" \
  "pycocotools==2.0.7"
```

Clone the official Grounding DINO repository, select the release commit used by
the Swin-T checkpoint, and install it without allowing its unpinned dependency
list to replace the Python 3.8-compatible packages:

```bash
mkdir -p ~/.local/share/go2_groundingdino
git clone https://github.com/IDEA-Research/GroundingDINO.git \
  ~/.local/share/go2_groundingdino/GroundingDINO
cd ~/.local/share/go2_groundingdino/GroundingDINO
python3 -m pip install --no-deps -e .

mkdir -p dino_weights
wget -O dino_weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

For CUDA execution, `CUDA_HOME` must refer to the CUDA toolkit used by the
installed PyTorch build before `pip install -e .` is run. JetPack normally
exposes it at `/usr/local/cuda`; verify and set it before installing Grounding
DINO if it is not already defined:

```bash
test -x /usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda
```

The official
[Grounding DINO installation instructions](https://github.com/IDEA-Research/GroundingDINO#hammer_and_wrench-install)
describe the CUDA extension build and `_C` extension troubleshooting.

Verify the environment before building the ROS package:

```bash
python3 -c "import numpy, torch, torchvision, groundingdino; print(numpy.__version__); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

On the JetPack 5.1.2 target this must report NumPy `1.23.5`, the Jetson-specific
PyTorch build, torchvision `0.16.2`, and CUDA availability as `True`. Do not
proceed with GPU deployment if the installed Torch build is CPU-only or CUDA is
unavailable.

When using a virtual environment, build with its Python interpreter so
the generated ROS executable points to that environment. If Grounding DINO was
installed into the existing system Python 3.8 environment, omit the activation
line but keep the explicit `python` invocation:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
source ~/.venvs/go2-groundingdino-py38/bin/activate
python3 "$(command -v colcon)" build \
  --packages-up-to intel_realsense_functions --symlink-install
source install/setup.bash
```

Invoking the `colcon` script through `python` is deliberate. It causes the
generated Python node wrapper to record the virtual environment's Python
3.8 interpreter instead of the system `colcon` script's shebang if a venv is active.

The interpreter recorded in the installed executable can be verified with:

```bash
head -n 1 \
  install/intel_realsense_functions/lib/intel_realsense_functions/groundingDinoNode
```

## Running the detector

Start the camera publisher and ensure that the robot TF tree includes the
`camera_link` and `map` frames. Start the detector with the Grounding DINO
configuration and checkpoint paths:

```bash
source /opt/ros/foxy/setup.bash
source ~/.venvs/go2-groundingdino-py38/bin/activate
source install/setup.bash

ros2 run intel_realsense_functions groundingDinoNode --ros-args \
  -p model_config_path:=$HOME/.local/share/go2_groundingdino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p model_checkpoint_path:=$HOME/.local/share/go2_groundingdino/GroundingDINO/dino_weights/groundingdino_swint_ogc.pth
```

The default `device` value is `auto`, which selects CUDA when PyTorch reports
that CUDA is available and otherwise selects the CPU. CPU inference is expected
to be substantially slower.

Monitor the results with:

```bash
ros2 topic echo /grounding_dino/detection_array
ros2 topic echo /grounding_dino/detections
ros2 run rqt_image_view rqt_image_view \
  /grounding_dino/annotated_image
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `model_config_path` | empty | Required Grounding DINO Python config path. |
| `model_checkpoint_path` | empty | Required model checkpoint path. |
| `device` | `auto` | Torch device: `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `box_threshold` | `0.35` | Minimum object-box score. |
| `text_threshold` | `0.25` | Minimum token score used to form a label. |
| `detection_rate_hz` | `5.0` | Maximum inference frequency. |
| `camera_frame` | `camera_link` | Robot camera TF frame for converted camera coordinates. |
| `map_frame` | `map` | TF frame for global coordinates. |
| `structured_detections_topic` | `/grounding_dino/detection_array` | Structured detection output topic. |
| `maximum_frame_age_sec` | `1.0` | Maximum accepted local receipt age. |
| `maximum_pair_offset_sec` | `0.25` | Maximum RGB/depth acquisition-time difference. |
| `minimum_depth_m` | `0.15` | Minimum valid depth. |
| `maximum_depth_m` | `6.0` | Maximum valid depth. |
| `depth_center_fraction` | `0.5` | Central box fraction used for depth sampling. |

All topic names are also parameters and may be remapped or overridden without
changing the source code.

## Coordinate and synchronization limitations

The current `getCameraFrames` node aligns depth to the RGB stream before
publishing, which permits direct use of the color intrinsics. It assigns the
same nonzero ROS timestamp to both images in an aligned pair without assigning
a frame ID. The detector pairs stamped images by acquisition time and requests
the `map <- camera_link` transform at that same time after converting optical
coordinates to the `camera_link` axis convention. For compatibility with other
camera publishers, zero-stamped images fall back to local receipt-time pairing
and the latest available transform. `maximum_frame_age_sec` still uses local
receipt age so delayed processing cannot make old images appear fresh.

The timestamp addition does not change the image topics, types, encodings,
dimensions, publication rate, or empty frame IDs. Existing consumers using
`message_filters.ApproximateTimeSynchronizer`, including the ArUco node, still
receive the same RGB/depth interface; equal pair timestamps make their
synchronization deterministic and allow TF lookup at the image time.
