# 📏 Benchmarking vS-Graphs

## 📊 Evaluation

To evaluate vS-Graphs against other Visual SLAM systems (e.g., ORB-SLAM3), follow the steps below:

### 🔧 1. Generate Pose Files

Prepare the `.txt` files containing estimated robot poses using the provided script:

- Run `generate_pose_txt_files.py` (from [here](../evaluation/generate_pose_txt_files.py)). Make sure that you set proper configuration parameters in [the config file](../evaluation/config.yaml). The script will create a `.txt` file and populate it with pose data.

### ▶️ 2. Run the SLAM Systems

Launch the system and play the `rosbag` file. Pose data will be recorded automatically during execution.

### 📈 3. Run Evaluation

Once both ground-truth and estimated poses are ready, use [`evo`](https://github.com/MichaelGrupp/evo) to compute Absolute Pose Error (APE):

```bash
evo_ape tum baseline_pose.txt slam_pose.txt -va > results.txt --plot --plot_mode xy
# Sample evo_ape tum s_graphs_pose_seq05.txt slam_pose_semuco_seq05.txt -va > results.txt --plot --plot_mode xy
```

## 🗺️ Working with Maps

vS-Graphs uses `.osa` files to store maps. These files are saved by default in your `ROS_HOME` directory (`~/.ros/`).

---

### 🔄 Loading a Map

- To load a saved map, set the `System.LoadAtlasFromFile` parameter in your SLAM settings YAML file.
- ⚠️ If no `.osa` file is available, **comment out** the `System.LoadAtlasFromFile` parameter to avoid runtime errors.

---

### 💾 Saving a Map

- **Option 1**: Enable `System.SaveAtlasToFile` in the SLAM settings file. The map will be automatically saved when you shut down the ROS node.
- **Option 2**: Manually trigger a save using the following ROS service:

```bash
rosservice call /vs_graphs/save_map [file_name]
```

### 🛠️ Map and Trajectory ROS Services

Use the following ROS services to save maps and trajectories during or after running vS-Graphs:

- To save the current SLAM map to `~/.ros/[file_name].osa` use `rosservice call /vs_graphs/save_map [file_name]`.
- To exports the estimated trajectories use `rosservice call /vs_graphs/save_traj [file_name]`
