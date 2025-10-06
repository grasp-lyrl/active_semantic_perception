# ⚙️ vS-Graphs Launch Arguments

This file documents the available _launch-time arguments_ that can be passed when launching `vS-Graphs`. Default values mainly represent **RealSense D435** camera settings.

## 📦 Default Launch Arguments

| Argument                    | Default Value                              | Description                                                           |
| --------------------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| `offline`                   | `true`                                     | Choose between **offline** (rosbag-based) and **live** sensor modes.  |
| `launch_rviz`               | `true`                                     | Launch `RViz` automatically with configured visual settings.          |
| `colored_pointcloud`        | `true`                                     | Apply artificial color for point clouds or use their real RGB values. |
| `visualize_segmented_scene` | `true`                                     | Toggle the visualization of segmented scenes in RViz.                 |
| `sensor_config`             | `RealSense_D435i`                          | Choose a predefined sensor configuration                              |
| `camera_frame`              | `camera`                                   | Set the reference `frame` name for the `camera`.                      |
| `rgb_image_topic`           | `/camera/color/image_raw`                  | Topic name for the RGB image stream.                                  |
| `rgb_camera_info_topic`     | `/camera/color/camera_info`                | Topic name for RGB camera intrinsics.                                 |
| `depth_image_topic`         | `/camera/aligned_depth_to_color/image_raw` | Topic name for the aligned depth image.                               |

To launch the system with default parameters, use the following command:

```bash
roslaunch vs_graphs vs_graphs.launch
```

## 📸 Live Mode - RealSense D435(i)

```bash
roslaunch vs_graphs vs_graphs.launch offline:=false
```

> 🛎️ Tip: Setting `offline` argument avoids **TF** conflicts by using live _RealSense TF_ instead of the one from the framework.

Next, launch the RealSense camera driver:

```bash
roslaunch realsense2_camera rs_rgbd.launch
# [Optional] align depth and IMU data arguments
# align_depth:=true unite_imu_method:=linear_interpolation
```

> 🛎️ Tip: To convert raw RealSense streams into usable depth images, you need to load nodelets via [rgbd_launch](http://wiki.ros.org/rgbd_launch). You can use a launch file similar to the one provided [here](/doc/RealSense/rs_d435_rgbd.launch).

> 🛎️ Tip: If you intend to use IR images instead of RGB, you will need to disable the RealSense **emitter**. While the launch file includes related arguments, they may not always take effect. The simplest method is to open `realsense-viewer` and manually set `Emitter Enabled` to `False`.

## 🧪 Using the ICL-NUIM Dataset

To use the [ICL dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) rosbags with `vS-Graphs`, set the appropriate parameters as shown below:

| Argument            | Default Value                              | New Value                 |
| ------------------- | ------------------------------------------ | ------------------------- |
| `sensor_config`     | `RealSense_D435i`                          | `ICL`                     |
| `depth_image_topic` | `/camera/aligned_depth_to_color/image_raw` | `/camera/depth/image_raw` |

Or simply launch:

```bash
roslaunch vs_graphs vs_graphs.launch sensor_config:=ICL depth_image_topic:=/camera/depth/image_raw
```

## 🧪 Using the OpenLoris Dataset

To use the [OpenLoris dataset](https://lifelong-robotic-vision.github.io/dataset/scene.html) rosbags with `vS-Graphs`, set the appropriate parameters as shown below:

| Argument                | Default Value                              | New Value                                |
| ----------------------- | ------------------------------------------ | ---------------------------------------- |
| `camera_frame`          | `camera`                                   | `d400_color`                             |
| `sensor_config`         | `RealSense_D435i`                          | `OpenLorisScene`                         |
| `rgb_image_topic`       | `/camera/color/image_raw`                  | `/d400/color/image_raw`                  |
| `rgb_camera_info_topic` | `/camera/color/camera_info`                | `/d400/color/camera_info`                |
| `depth_image_topic`     | `/camera/aligned_depth_to_color/image_raw` | `/d400/aligned_depth_to_color/image_raw` |

Or simply launch:

```bash
roslaunch vs_graphs vs_graphs.launch sensor_config:=OpenLorisScene rgb_image_topic:=/d400/color/image_raw depth_image_topic:=/d400/aligned_depth_to_color/image_raw rgb_camera_info_topic:=/d400/color/camera_info camera_frame:=d400_color
```

## 🧪 Using the ScanNet Dataset

To use the [ScanNet dataset](http://www.scan-net.org/) rosbags with `vS-Graphs`, set the appropriate parameters as shown below:

| Argument            | Default Value                              | New Value                 |
| ------------------- | ------------------------------------------ | ------------------------- |
| `sensor_config`     | `RealSense_D435i`                          | `ScanNet`                 |
| `depth_image_topic` | `/camera/aligned_depth_to_color/image_raw` | `/camera/depth/image_raw` |

Or simply launch:

```bash
roslaunch vs_graphs vs_graphs.launch sensor_config:=ScanNet depth_image_topic:=/camera/depth/image_raw
```

## 🧪 Using the TUM RGB-D Dataset

To use the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) rosbags with `vS-Graphs`, set the appropriate parameters as shown below:

| Argument                | Default Value                              | New Value                 |
| ----------------------- | ------------------------------------------ | ------------------------- |
| `camera_frame`          | `camera`                                   | `kinect`                  |
| `sensor_config`         | `RealSense_D435i`                          | `TUM1` / `TUM2` / `TUM3`  |
| `rgb_image_topic`       | `/camera/color/image_raw`                  | `/camera/rgb/image_color` |
| `depth_image_topic`     | `/camera/aligned_depth_to_color/image_raw` | `/camera/depth/image`     |
| `rgb_camera_info_topic` | `/camera/color/camera_info`                | `/camera/rgb/camera_info` |

Or simply launch:

```bash
roslaunch vs_graphs vs_graphs.launch rgb_image_topic:=/camera/rgb/image_color depth_image_topic:=/camera/depth/image rgb_camera_info_topic:=/camera/rgb/camera_info sensor_config:=TUM3 camera_frame:=kinect
```
