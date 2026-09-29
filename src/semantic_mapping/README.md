# Semantic mapping

`semantic_mapping` converts Grounding DINO observations into a stateful map of
objects in the ROS `map` frame. It assigns stable object identities, fuses
repeated position estimates, tracks object age and movement, and publishes an
RViz representation.

The package does not run object detection and does not control the robot. It
consumes the structured detections published by
`intel_realsense_functions/groundingDinoNode`.

## Interfaces

| Direction | Topic | Type | Description |
| --- | --- | --- | --- |
| Input | `/grounding_dino/detection_array` | `intel_realsense_interfaces/msg/GroundedDetectionArray` | Grounded detections with optional map positions. |
| Output | `/semantic_map` | `intel_realsense_interfaces/msg/SemanticMap` | Confirmed semantic objects and their lifecycle state. |
| Output | `/semantic_map/markers` | `visualization_msgs/msg/MarkerArray` | Object spheres and labels for RViz. |

Both output publishers use reliable, transient-local QoS. A late subscriber
therefore receives the latest map without waiting for another detection.

Each `SemanticObject` contains:

- A stable `object_id` such as `object_000001`.
- Its semantic `label` and fused map-frame `position`.
- Fused confidence and total observation count.
- First- and last-observed timestamps.
- The number of confirmed movements.
- An `ACTIVE`, `STALE`, or `REMOVED` status.

Tentative tracks are intentionally not published.

The map is currently held in memory. Object IDs remain stable across
observations, movement, stale periods, and removal transitions while the node
is running. Restarting the node starts a new map and resets the ID sequence;
disk persistence can be added later once map persistence and map-frame reuse
are defined for the navigation stack.

## Association and duplicate prevention

Association is performed independently for each semantic label. Labels are
lowercased and internal whitespace is normalized for comparison, while the
original label is retained for display.

The node applies the following stages to each detection message:

1. Detections without map coordinates or below `minimum_confidence` are
   rejected. This prevents an uncertain pixel detection from entering the map.
2. Same-label detections within `observation_merge_distance_m` in the same
   message are collapsed into one confidence-weighted observation. This handles
   overlapping boxes produced for the same physical object.
3. Remaining observations are processed from highest confidence to lowest.
   Each is matched to the nearest unmatched track with the same normalized
   label inside `association_distance_m`. A track can be matched at most once
   per detection message.
4. An unmatched observation creates a tentative track. The track is not
   published until it reaches `confirmation_observations`. A one-frame false
   positive therefore never becomes part of the semantic map. Tentative tracks
   that are not reinforced within `tentative_timeout_sec` are discarded.
5. Confirmed tracks retain their object ID for their lifetime. New position
   measurements inside the normal movement threshold are fused with a
   confidence-scaled exponential moving average rather than replacing the
   stored position. This reduces depth and TF jitter.

This is nearest-neighbor data association rather than visual re-identification.
Two same-label objects closer than the configured spatial gates can still be
ambiguous. The gates should therefore be selected for the expected localization
noise and object spacing.

## Moved and removed objects

A single displaced detection cannot immediately move a mapped object. When an
associated measurement is farther than `movement_distance_m`, the node starts a
movement candidate. The stored position changes only after
`movement_confirmation_observations` mutually consistent measurements inside
`movement_cluster_distance_m`. The object keeps the same identity and its
`movement_count` increases.

If an observation is outside the normal association gate, it first forms a new
tentative track. Once confirmed, it is treated as a relocation of an older
same-label object only when exactly one eligible old object exists, it has not
been observed for `relocation_minimum_age_sec`, and the new position is within
`relocation_maximum_distance_m`. Requiring one unambiguous candidate avoids
silently transferring an identity when several cups or similar objects exist.

A confirmed object becomes `STALE` after `stale_after_sec` without an
observation and `REMOVED` after `removed_after_sec`. Removed objects remain in
the semantic-map message for `removed_retention_sec` so consumers can observe
the state transition, while their RViz markers are deleted immediately. After
the retention interval, the record is pruned.

Removal is necessarily time-based: not observing an object does not prove that
it was removed if the camera is pointed elsewhere or detection is stopped. The
default five-minute removal timeout is intentionally conservative. A future
scene-coverage component can provide stronger negative evidence.

## Build and run

Build the interface package and mapping package with their dependencies:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
python3 "$(command -v colcon)" build \
  --packages-up-to semantic_mapping --symlink-install
source install/setup.bash
```

Start Grounding DINO and semantic mapping together:

```bash
ros2 launch semantic_mapping semantic_mapping.launch.py
```

This launch file intentionally does not start `getCameraFrames`. The camera
publisher and the TF publishers must already be running, whether they were
started individually or by the main robot launch system.

The default model paths match the Grounding DINO installation documented by
`intel_realsense_functions`:

```text
~/.local/share/go2_groundingdino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
~/.local/share/go2_groundingdino/GroundingDINO/dino_weights/groundingdino_swint_ogc.pth
```

Paths and shared frame settings can be overridden on the launch command:

```bash
ros2 launch semantic_mapping semantic_mapping.launch.py \
  model_config_path:=/path/to/GroundingDINO_SwinT_OGC.py \
  model_checkpoint_path:=/path/to/groundingdino_swint_ogc.pth \
  camera_frame:=front_camera \
  map_frame:=map
```

The launch file exposes the following arguments:

| Launch argument | Default | Applied to |
| --- | --- | --- |
| `model_config_path` | Standard user-local Grounding DINO path | Detector |
| `model_checkpoint_path` | Standard user-local checkpoint path | Detector |
| `device` | `auto` | Detector |
| `detection_rate_hz` | `5.0` | Detector |
| `camera_frame` | `front_camera` | Detector |
| `map_frame` | `map` | Both nodes |
| `detections_topic` | `/grounding_dino/detection_array` | Both nodes |
| `minimum_confidence` | `0.50` | Semantic mapper |
| `confirmation_observations` | `3` | Semantic mapper |

The detector and mapper receive the same `map_frame` and structured detection
topic from the launch file so the connection cannot drift through separate
configuration. To run only the mapper for debugging, use:

```bash
ros2 run semantic_mapping semanticMappingNode
```

Inspect the structured map:

```bash
ros2 topic echo /semantic_map
```

In RViz, set the fixed frame to `map`, add a `MarkerArray` display, and select
`/semantic_map/markers`. Active objects are green. Stale objects are gray, and
removed objects are deleted from the display.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `detections_topic` | `/grounding_dino/detection_array` | Structured detection input. |
| `semantic_map_topic` | `/semantic_map` | Semantic map output. |
| `markers_topic` | `/semantic_map/markers` | RViz marker output. |
| `map_frame` | `map` | Required coordinate frame for input positions and outputs. |
| `publish_rate_hz` | `1.0` | Periodic map and marker publication rate. |
| `minimum_confidence` | `0.50` | Lowest detection score accepted by the mapper. |
| `confirmation_observations` | `3` | Observations required before publishing an identity. |
| `observation_merge_distance_m` | `0.20` | Same-frame duplicate merge radius. |
| `association_distance_m` | `0.75` | Normal same-label track association radius. |
| `position_fusion_alpha` | `0.25` | Maximum exponential position update weight. |
| `movement_distance_m` | `0.35` | Displacement that starts movement confirmation. |
| `movement_cluster_distance_m` | `0.30` | Consistency radius for displaced observations. |
| `movement_confirmation_observations` | `3` | Consistent observations required to accept movement. |
| `relocation_minimum_age_sec` | `5.0` | Minimum unseen time before far re-identification. |
| `relocation_maximum_distance_m` | `3.0` | Maximum far re-identification distance. |
| `tentative_timeout_sec` | `10.0` | Lifetime of an unconfirmed candidate. |
| `stale_after_sec` | `30.0` | Unseen time before an object becomes stale. |
| `removed_after_sec` | `300.0` | Unseen time before an object becomes removed. |
| `removed_retention_sec` | `300.0` | Time to retain a removed record before pruning. |
| `include_removed_objects` | `true` | Include retained removed records in `/semantic_map`. |
| `marker_scale_m` | `0.18` | RViz object marker diameter. |
