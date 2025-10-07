#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import message_filters
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose
import numpy as np
import cv2
import math
import os
import time
import quaternion
from dataclasses import dataclass
from pynput import keyboard
import habitat_sim
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
import shutil
import json
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from local_planning import LocalPlanner
from mid_planning import MidPlanner
import yaml


@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near_clip: float
    max_range: float


@dataclass
class PipelineConfig:
    SCENE_DATASET_CONFIG: str
    SCENE_ID: str
    SCENE_NUMBER: int
    AGENT_HEIGHT: float
    RGB_TOPIC: str
    DEPTH_TOPIC: str
    MID_PATH_TOPIC: str
    LOCAL_PATH_TOPIC: str
    WORLD_FRAME: str
    HABITAT_FRAME: str
    AGENT_FRAME: str
    MERGED_OCCUPANCY_GRID_TOPIC: str
    OCCUPANCY_GRAPH_COUNT: int
    OCCUPANCY_GRID_TOPIC_PREFIX: str
    INFLATION_RADIUS: float
    BASE_DIRECTORY: str
    MAPPING_DIRECTORY: str
    SCENE_GRAPH_FILENAMES: list[str]
    FILE_WAIT_TIMEOUT_S: int
    MIN_DSG_FILE_SIZE_BYTES: int
    TURN_STEP_DEG: float
    MOVE_STEP_M: float
    DISTANCE_THRESHOLD_M: float
    TURN_THRESHOLD_RAD: float
    MINIMUM_NAVIGATION_DURATION: float
    UNKNOWN_CELL_THRESHOLD: float
    MAXIMUM_CORNER_ATTEMPTS: int
    CAMERA_CONFIG: CameraConfig

# ## -----------------------------------------------------------------------------
# ## Pipeline Configuration
# ## -----------------------------------------------------------------------------
# @dataclass(frozen=True)
# class CameraConfig:
#     fx: float = 320.0
#     fy: float = 240.0
#     cx: float = 319.5
#     cy: float = 239.5
#     width: int = 640
#     height: int = 480
#     near_clip: float = 0.5
#     max_range: float = 3.0

# class PipelineConfig:
#     """Central configuration for the entire pipeline."""

#     # --- Habitat Simulation ---
#     SCENE_DATASET_CONFIG = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
#     # 00871
#     SCENE_ID = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/val/00871-VBzV5z6i1WS/VBzV5z6i1WS.basis.glb'
#     # 00853
#     # SCENE_ID = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
#     # 00573
#     # SCENE_ID = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00573-1zDbEdygBeW/1zDbEdygBeW.basis.glb"
#     AGENT_HEIGHT = 1.1

#     # --- ROS Topics & Frames ---
#     RGB_TOPIC = '/dominic/forward/color/image_raw'
#     DEPTH_TOPIC = '/dominic/forward/depth/image_rect_raw'
#     MID_PATH_TOPIC = '/planned_mid_path'
#     LOCAL_PATH_TOPIC = '/planned_local_path'
#     WORLD_FRAME = 'world'
#     HABITAT_FRAME = 'world_habitat'
#     AGENT_FRAME = 'dominic/forward_link'

#     # --- Occupancy Grid Configuration ---
#     MERGED_OCCUPANCY_GRID_TOPIC = '/gvd_merged_map'
#     OCCUPANCY_GRAPH_COUNT = 2
#     OCCUPANCY_GRID_TOPIC_PREFIX = '/clio_node/graph'
#     INFLATION_RADIUS = 0.15

#     BASE_DIRECTORY = '/home/apple/Work/scene_graph_processing/Summer_research/Frontier_00871'
#     MAPPING_DIRECTORY = '/home/apple/Work/exp_pipeline_ws/src/hydra/output/realsense/backend'
#     SCENE_GRAPH_FILENAMES = ['graph0_dsg.json', 'graph1_dsg.json']
#     FILE_WAIT_TIMEOUT_S = 10
#     MIN_DSG_FILE_SIZE_BYTES = 1000

#     # --- Navigation Parameters ---
#     TURN_STEP_DEG = 1.5
#     MOVE_STEP_M = 0.075
#     DISTANCE_THRESHOLD_M = 0.25
#     TURN_THRESHOLD_RAD = math.radians(10.0)
#     MINIMUM_NAVIGATION_DURATION = 13.0
#     UNKNOWN_CELL_THRESHOLD = 0.4
#     MAXIMUM_CORNER_ATTEMPTS = 5

#     # --- Camera Configuration ---
#     CAMERA_CONFIG = CameraConfig()

## -----------------------------------------------------------------------------
## Utility Functions
## -----------------------------------------------------------------------------
def get_around_x_matrix(angle):
    return np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]])

def get_around_z_matrix(angle):
    return np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0,0,0,1]])

def transform_habitat_pose_to_z_up(agent_s):
    rot_q=get_around_x_matrix(-np.pi/2)@get_around_z_matrix(np.pi/4)
    homogeneous_matrix=np.eye(4); homogeneous_matrix[:3,:3]=quaternion.as_rotation_matrix(agent_s.rotation); homogeneous_matrix[:3,3]=agent_s.position
    new_matrix=homogeneous_matrix@rot_q
    return new_matrix[:3,3], quaternion.from_rotation_matrix(new_matrix[:3,:3])

def transform_habitat_pose_to_z_front(agent_s):
    rot_q=get_around_x_matrix(-np.pi)
    homogeneous_matrix=np.eye(4); homogeneous_matrix[:3,:3]=quaternion.as_rotation_matrix(agent_s.rotation); homogeneous_matrix[:3,3]=agent_s.position
    new_matrix=homogeneous_matrix@rot_q
    return new_matrix[:3,3], quaternion.from_rotation_matrix(new_matrix[:3,:3])


class FrontierPlanner:
    """Finds the closest frontier on a merged occupancy grid."""
    def __init__(self, config):
        self.visited_frontiers = []
        self.config = config
        self.occupancy_grid = None
        self.grid_info = None
        self.grid_sub = rospy.Subscriber(self.config.MERGED_OCCUPANCY_GRID_TOPIC, OccupancyGrid, self.grid_callback)
        self.cluster_marker_pub = rospy.Publisher('/frontier_clusters', Marker, queue_size=10)

    def grid_callback(self, msg):
        self.grid_info = msg.info
        self.occupancy_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def world_to_grid(self, world_pos):
        if not self.grid_info: return None
        grid_x = int((world_pos[0] - self.grid_info.origin.position.x) / self.grid_info.resolution)
        grid_y = int((world_pos[1] - self.grid_info.origin.position.y) / self.grid_info.resolution)
        if 0 <= grid_x < self.grid_info.width and 0 <= grid_y < self.grid_info.height:
            return (grid_x, grid_y)
        return None

    def grid_to_world(self, grid_pos):
        world_x = (grid_pos[0] + 0.5) * self.grid_info.resolution + self.grid_info.origin.position.x
        world_y = (grid_pos[1] + 0.5) * self.grid_info.resolution + self.grid_info.origin.position.y
        return np.array([world_x, world_y]) # Assume exploration is planar in world frame
    
    def _publish_cluster_markers(self, frontiers_grid_coords, cluster_labels):
        """Publishes clustered frontier points as a Marker for RViz, with a different color for each cluster."""
        marker = Marker()
        marker.header.frame_id = self.config.WORLD_FRAME
        marker.header.stamp = rospy.Time.now()
        marker.ns = "frontier_clusters"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.06
        marker.scale.y = 0.06

        marker.points = []
        marker.colors = []

        color_palette = [
            # Reds
            ColorRGBA(0.9, 0.2, 0.2, 1.0),    # Bright Red
            ColorRGBA(0.6, 0.1, 0.1, 1.0),    # Dark Red

            # Oranges
            ColorRGBA(1.0, 0.6, 0.2, 1.0),    # Bright Orange
            ColorRGBA(0.8, 0.4, 0.0, 1.0),    # Dark Orange

            # Yellows
            ColorRGBA(1.0, 0.9, 0.3, 1.0),    # Bright Yellow
            ColorRGBA(0.7, 0.6, 0.1, 1.0),    # Dark Yellow (Mustard)

            # Greens
            ColorRGBA(0.4, 0.9, 0.4, 1.0),    # Bright Green
            ColorRGBA(0.1, 0.5, 0.1, 1.0),    # Dark Green

            # Cyans
            ColorRGBA(0.3, 0.9, 0.9, 1.0),    # Bright Cyan
            ColorRGBA(0.1, 0.5, 0.5, 1.0),    # Dark Cyan (Teal)

            # Blues
            ColorRGBA(0.2, 0.5, 1.0, 1.0),    # Bright Blue
            ColorRGBA(0.1, 0.2, 0.6, 1.0),    # Dark Blue (Navy)

            # Violets
            ColorRGBA(0.6, 0.4, 1.0, 1.0),    # Bright Violet
            ColorRGBA(0.3, 0.1, 0.6, 1.0),    # Dark Violet (Indigo)

            # Magentas
            ColorRGBA(0.9, 0.2, 0.9, 1.0),    # Bright Magenta
            ColorRGBA(0.6, 0.0, 0.6, 1.0),    # Dark Magenta

            # Pinks
            ColorRGBA(1.0, 0.4, 0.6, 1.0),    # Bright Pink
            ColorRGBA(0.8, 0.2, 0.4, 1.0),    # Dark Pink (Rose)
            
            # Browns
            ColorRGBA(0.8, 0.65, 0.45, 1.0),  # Light Brown (Tan)
            ColorRGBA(0.5, 0.3, 0.1, 1.0),    # Dark Brown
        ]
        
        noise_color = ColorRGBA(0.5, 0.5, 0.5, 0.5)

        for point_grid, label in zip(frontiers_grid_coords, cluster_labels):
            world_p = self.grid_to_world(point_grid)
            p = Point(world_p[0], world_p[1], 0)
            marker.points.append(p)

            if label == -1:
                marker.colors.append(noise_color)
            else:
                color_index = label % len(color_palette)
                marker.colors.append(color_palette[color_index])

        self.cluster_marker_pub.publish(marker)

    def find_frontiers(self, grid):
        h, w = grid.shape
        frontiers = set()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 0:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == -1:
                                frontiers.add((x, y))
                                break
        return np.array(list(frontiers))

    def plan(self, current_world_pos):
        if self.occupancy_grid is None:
            rospy.logwarn("Frontier planner has no map yet.")
            return None

        frontiers = self.find_frontiers(self.occupancy_grid)
        eps = 1
        min_samples = 3
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(frontiers)
        self._publish_cluster_markers(frontiers, cluster_labels)
        poses_to_explore = []
        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_points = frontiers[cluster_indices]

            world_cluster = np.array([self.grid_to_world(point) for point in cluster_points])
            closest_point = world_cluster[np.argmin(np.linalg.norm(world_cluster - current_world_pos[:2].reshape(1, -1), axis=1))]
            cp = [closest_point[0], closest_point[1]]
            is_visited = any(np.linalg.norm(np.array(cp) - np.array(visited_point)) < 0.4 for visited_point in self.visited_frontiers)
            if is_visited:
                continue
            poses_to_explore.append(np.array([closest_point[0], closest_point[1], self.config.AGENT_HEIGHT]))

        # It means all the frontiers have been explored
        if len(poses_to_explore) == 0:
            return None

        poses_to_explore = sorted(poses_to_explore, key=lambda x: np.linalg.norm(x - current_world_pos))
        target_world_pos = poses_to_explore[0]
        self.visited_frontiers.append([target_world_pos[0], target_world_pos[1]])
        return target_world_pos


class FrontierPipeline:
    def __init__(self, config):
        self.config = config
        self.state = "IDLE"  # States: IDLE, SCANNING, PLANNING, NAVIGATING, FINISHED
        self.stop_requested = False

        rospy.init_node('complete_frontier_baseline', anonymous=True)
        self.bridge = CvBridge(); self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rgb_pub = rospy.Publisher(self.config.RGB_TOPIC, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(self.config.DEPTH_TOPIC, Image, queue_size=10)
        self.mid_path_sub = rospy.Subscriber(self.config.MID_PATH_TOPIC, Path, self.mid_path_callback)
        self.local_path_sub = rospy.Subscriber(self.config.LOCAL_PATH_TOPIC, Path, self.local_path_callback)
        self.rate = rospy.Rate(10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.frontier_planner = FrontierPlanner(config)

        self.mid_planner = MidPlanner(config, self.tf_buffer)
        self.local_planner = LocalPlanner(config, self.tf_buffer)

        # --- State & Sim Variables ---
        self.previous_dsg_sizes = {}
        self.current_target_index = -1
        self.stage = 0
        self.total_path_length = 0.0
        self.total_navigation_time = 0.0
        self.exploration_start_time = None
        # Exact path saving (both in world frame and habitat frame)
        self.current_stage_path_habitat = []
        self.current_stage_path_world = []

        # --- Habitat Simulation Setup ---
        self.sim = self._initialize_simulator()
        self.pathfinder = self.sim.pathfinder
        self.agent = self.sim.initialize_agent(0)
        self.agent_initial_state = self.agent.get_state()
        initial_state = self.agent.get_state()
        self.navigation_start_time = None
        self.local_path = None
        self.history_position = []
        self.corner_attempts = 0
        if self.config.SCENE_NUMBER == 871:
            # 00871
            initial_state.position = np.array([-5.5, self.config.AGENT_HEIGHT, -3.0])
        elif self.config.SCENE_NUMBER == 853:
            # 00853
            initial_state.position = np.array([-0.5, self.config.AGENT_HEIGHT, 4.5])
        elif self.config.SCENE_NUMBER == 573:
            # 00573
            initial_state.position = np.array([0.1, self.config.AGENT_HEIGHT, 1.5])
        self.agent.set_state(initial_state)

    def mid_path_callback(self, msg):
        """Receives the mid path and prepares for navigation."""
        if len(msg.poses) > 0:
            mid_path = []
            for pose in msg.poses:
                point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                mid_path.append(point)
            self.local_planner.set_mid_level_path(mid_path)

    def _record_agent_position(self):
        """Records the agent's current position in both frames."""
        current_pos_habitat = self.agent.get_state().position
        
        # Avoid recording duplicate points if the agent only turns
        if np.array_equal(self.current_stage_path_habitat[-1], current_pos_habitat):
            return

        current_pos_world = self._transform_pose_to_world(current_pos_habitat)

        self.current_stage_path_habitat.append(current_pos_habitat.tolist())
        self.current_stage_path_world.append(current_pos_world.tolist())

    def local_path_callback(self, msg):
        """Receives the local path and prepares for navigation."""
        if not msg.poses:
            rospy.loginfo("Local planning failed. Transitioning to SCANNING.")
            navigation_duration_s = rospy.get_time() - self.navigation_start_time
            self.total_navigation_time += navigation_duration_s
            self.mid_planner.set_high_level_path(None)
            self.local_planner.set_mid_level_path(None)
            self.local_path = None
            self.current_target_index = -1
            self.state = "SCANNING"
            self.stage += 1
            next_stage_directory = os.path.join(self.config.BASE_DIRECTORY, str(self.stage))
            os.makedirs(next_stage_directory, exist_ok=True)
            path_filepath = os.path.join(next_stage_directory, "executed_path.json")
            path_data = {
                "habitat_frame_path": self.current_stage_path_habitat,
                "world_frame_path": self.current_stage_path_world
            }
            with open(path_filepath, "w") as json_file:
                json.dump(path_data, json_file)
            return

        # Transform the path from ROS message to Habitat coordinates
        global_path = []
        for pose in msg.poses:
            point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            global_path.append(point)
        self.local_path = self._transform_path_to_habitat(global_path, self.tf_buffer)
        # A fix to make navigation faster
        # Find the closest index
        current_pos = self.agent.get_state().position
        min_dist = float('inf')
        closest_index = 0
        for i, waypoint in enumerate(self.local_path):
            delta_pos = waypoint - current_pos
            delta_pos[1] = 0
            dist = np.linalg.norm(delta_pos)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
                
        # Find the first index that is outside navigation radius
        target_index = closest_index
        for i in range(closest_index, len(self.local_path)):
            waypoint = self.local_path[i]
            delta_pos = waypoint - current_pos
            delta_pos[1] = 0
            dist = np.linalg.norm(delta_pos)
            
            if dist > self.config.DISTANCE_THRESHOLD_M:
                target_index = i
                break

        # If the loop completes without breaking, it means all remaining points are
        # within the threshold. In this case, targeting the last point is a safe fallback.
        if target_index == closest_index and min_dist < self.config.DISTANCE_THRESHOLD_M:
            target_index = len(self.local_path) - 1

        self.current_target_index = target_index

    def _transform_path_to_habitat(self, path, tf_buffer):
        transformed_path = []
        for point in path:
            raw_pose = PoseStamped()
            raw_pose.header.frame_id = "world"
            raw_pose.header.stamp = rospy.Time(0)
            raw_pose.pose.position.x = point[0]
            raw_pose.pose.position.y = point[1]
            raw_pose.pose.position.z = point[2]

            target_frame = "world_habitat"
            transformed_pose_stamped = None

            try:
                transformed_pose_stamped = tf_buffer.transform(
                    raw_pose,
                    target_frame,
                    timeout=rospy.Duration(1.0)
                )

            except tf2_ros.LookupException:
                rospy.logerr("Transform not found. Waiting for transform...")

            if transformed_pose_stamped is not None:
                position = np.array([
                    transformed_pose_stamped.pose.position.x,
                    transformed_pose_stamped.pose.position.y,
                    transformed_pose_stamped.pose.position.z])
                transformed_path.append(position)

        return transformed_path

    def _initialize_simulator(self):
        """Sets up the Habitat simulator configuration."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = self.config.SCENE_DATASET_CONFIG
        sim_cfg.scene_id = self.config.SCENE_ID
        sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        # Sensor specifications
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [480, 640]
        depth_sensor_spec.position = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    def _transform_point(self, input_point, source_frame, target_frame):
        raw_pose = PoseStamped()
        raw_pose.header.frame_id = source_frame
        raw_pose.header.stamp = rospy.Time(0)
        raw_pose.pose.position.x = input_point[0]
        raw_pose.pose.position.y = input_point[1]
        raw_pose.pose.position.z = input_point[2]
        raw_pose.pose.orientation.w = 1.0  # Neutral orientation

        try:
            transformed_pose = self.tf_buffer.transform(
                raw_pose,
                target_frame,
                timeout=rospy.Duration(1.0)
            )

            return np.array([
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z
            ])

        except tf2_ros.LookupException:
            rospy.logerr("Transform not found. Waiting for transform...")
            return None
        
    def _transform_pose_to_world(self, habitat_pose):
        """Transforms a pose from the habitat frame to the world frame."""
        return self._transform_point(habitat_pose, "world_habitat", "world")

    def _transform_pose_to_habitat(self, world_pose):
        """Transforms a pose from the world frame to the habitat frame."""
        return self._transform_point(world_pose, "world", "world_habitat")

    def publish_tfs(self):
        # Habitat to World TF
        pos, orient = transform_habitat_pose_to_z_up(self.agent_initial_state)
        t=TransformStamped();t.header.stamp=rospy.Time.now();t.header.frame_id=self.config.HABITAT_FRAME;t.child_frame_id=self.config.WORLD_FRAME
        t.transform.translation.x=pos[0];t.transform.translation.y=pos[1];t.transform.translation.z=pos[2]
        t.transform.rotation.x=orient.x;t.transform.rotation.y=orient.y;t.transform.rotation.z=orient.z;t.transform.rotation.w=orient.w
        self.tf_broadcaster.sendTransform(t)
        # Agent to Habitat TF
        agent_state = self.agent.get_state()
        pos, orient = transform_habitat_pose_to_z_front(agent_state)
        t=TransformStamped();t.header.stamp=rospy.Time.now();t.header.frame_id=self.config.HABITAT_FRAME;t.child_frame_id=self.config.AGENT_FRAME
        t.transform.translation.x=pos[0];t.transform.translation.y=pos[1];t.transform.translation.z=pos[2]
        t.transform.rotation.w=orient.w;t.transform.rotation.x=orient.x;t.transform.rotation.y=orient.y;t.transform.rotation.z=orient.z
        self.tf_broadcaster.sendTransform(t)

    def publish_sensor_images(self):
        observations=self.sim.get_sensor_observations();timestamp=rospy.Time.now()
        rgb_image=cv2.cvtColor(observations['color_sensor'],cv2.COLOR_RGBA2RGB)
        rgb_msg=self.bridge.cv2_to_imgmsg(rgb_image,encoding='rgb8');rgb_msg.header.stamp=timestamp;rgb_msg.header.frame_id=self.config.AGENT_FRAME
        self.rgb_pub.publish(rgb_msg)
        depth_image=observations['depth_sensor']
        depth_msg=self.bridge.cv2_to_imgmsg(depth_image,encoding='32FC1');depth_msg.header.stamp=timestamp;depth_msg.header.frame_id=self.config.AGENT_FRAME
        self.depth_pub.publish(depth_msg)

    def _agent_turn(self, degrees):
        state=self.agent.get_state();rot=habitat_sim.utils.common.quat_from_angle_axis(math.radians(degrees),np.array([0,1,0]));state.rotation=rot*state.rotation
        self.agent.set_state(state)

    def _agent_forward(self, dist):
        state=self.agent.get_state();forward_vector=quaternion.as_rotation_matrix(state.rotation)@np.array([0,0,-1]);forward_vector[1]=0
        state.position+=forward_vector*dist
        self.agent.set_state(state)
        self.total_path_length += dist

    def execute_scanning(self):
        rospy.loginfo("State: SCANNING - Performing 360-degree scan.")
        for _ in range(int(360 / self.config.TURN_STEP_DEG)):
            if rospy.is_shutdown(): break
            self._agent_turn(self.config.TURN_STEP_DEG)
            self.publish_tfs()
            self.publish_sensor_images()
            self.rate.sleep()
        self.state = "PLANNING"

    def execute_planning(self):
        rospy.loginfo("State: PLANNING - Looking for frontiers.")
        current_habitat_pos = self.agent.get_state().position
        current_world_pos = self._transform_pose_to_world(current_habitat_pos)
        if current_world_pos is None:
            rospy.logerr("Could not get agent position. Retrying scan.")
            self.state = "SCANNING"
            return

        # 1. Get the single goal point from the Frontier Planner
        target_world_pos = self.frontier_planner.plan(current_world_pos)
        if target_world_pos is None:
            rospy.logwarn("Frontier planner returned no target. Exploration finished.")
            self.state = "FINISHED"
            return

        target_habitat_pos = self._transform_pose_to_habitat(target_world_pos)
        # This avoids issues if the target is in an unreachable location
        navigable_habitat_pos = self.pathfinder.snap_point(target_habitat_pos)
        if navigable_habitat_pos is not None:
            target_habitat_pos = navigable_habitat_pos
            target_world_pos = self._transform_pose_to_world(target_habitat_pos)
        else:
            # Let the planner handle it
            rospy.logwarn("Pathfinder could not find a path to the target. Proceed to original target.")
            
        # 2. Create a simple high-level path (start -> goal) for the MidPlanner
        high_level_path = [current_world_pos, target_world_pos]
        self.mid_planner.set_high_level_path(high_level_path)

        # 3. Trigger the mid-planner to start planning once.
        self.mid_planner.plan_path()

        # 4. Store all the results from this stage
        self._archive_results()
        working_directory = os.path.join(self.config.BASE_DIRECTORY, str(self.stage))
        json_filepath = os.path.join(working_directory, "navigation_stats.json")
        with open(json_filepath, "w") as json_file:
            json.dump({"total_path_length_meters": self.total_path_length,
            "total_navigation_time_seconds": self.total_navigation_time}, json_file)

        # 5. Start recording the path for this stage
        self.current_stage_path_habitat = []
        self.current_stage_path_world = []
        start_pos_habitat = self.agent.get_state().position
        start_pos_world = self._transform_pose_to_world(start_pos_habitat)
        self.current_stage_path_habitat.append(start_pos_habitat.tolist())
        self.current_stage_path_world.append(start_pos_world.tolist())

        # 6. Transition to navigation. The callbacks will handle populating the local path.
        self.corner_attempts = 0
        self.navigation_start_time = rospy.get_time()
        self.state = "NAVIGATING"

    # Attempts to nudge the robot forward if it's stuck but not at the final goal. This technique is a workaround due to the limited scope of local planner.
    def _execute_corner_recovery(self):
        if self.corner_attempts >= self.config.MAXIMUM_CORNER_ATTEMPTS:
            rospy.logwarn(f"Corner recovery failed: maximum of {self.config.MAXIMUM_CORNER_ATTEMPTS} attempts reached.")
            return False

        if not self.local_planner.mid_level_path:
            return False
        
        final_goal_world = self.local_planner.mid_level_path[-1]
        current_pos_world = self._transform_pose_to_world(self.agent.get_state().position)

        dist_to_final_goal = np.linalg.norm((final_goal_world - current_pos_world)[:2])
        if dist_to_final_goal < 0.5:
            return False

        rospy.loginfo("Local goal reached, but far from global goal. Attempting 'nudge' recovery.")
        nudge_dist = 0.1
        relative_nudges = [
            ("straight", np.array([0, 0, -nudge_dist])),
            ("diag_right", np.array([-nudge_dist * 0.707, 0, -nudge_dist * 0.707])),
            ("diag_left", np.array([nudge_dist * 0.707, 0, -nudge_dist * 0.707]))
        ]
        
        agent_state = self.agent.get_state()
        rot_matrix = quaternion.as_rotation_matrix(agent_state.rotation)

        for direction_name, rel_pos in relative_nudges:
            nudge_habitat_world = agent_state.position + rot_matrix @ rel_pos
            nudge_ros_world = self._transform_pose_to_world(nudge_habitat_world)

            grid_coords = self.local_planner.world_to_grid(nudge_ros_world[0], nudge_ros_world[1])
            if grid_coords is not None:
                grid_x, grid_y = grid_coords
                # Check if the single target cell is free (value 0)
                target_index = grid_y * self.local_planner.current_grid.info.width + grid_x
                if self.local_planner.current_grid.data[target_index] == 0:
                    rospy.loginfo(f"Nudge successful (Attempt {self.corner_attempts + 1}/{self.config.MAXIMUM_CORNER_ATTEMPTS}): moving {direction_name}.")
                    self.corner_attempts += 1
                    
                    new_state = self.agent.get_state()
                    new_state.position = nudge_habitat_world
                    self.agent.set_state(new_state)
                    self._record_agent_position()
                    self.total_path_length += nudge_dist
                    
                    self.local_path = None
                    self.current_target_index = -1
                    return True

        rospy.logwarn(f"Corner recovery failed on attempt {self.corner_attempts + 1}/{self.config.MAXIMUM_CORNER_ATTEMPTS}. All three forward directions are blocked.")
        return False


    def execute_navigation(self):
        if not self.local_path:
            self.current_target_index = -1
            self._agent_turn(self.config.TURN_STEP_DEG * 2)
            return
        
        if self.current_target_index >= len(self.local_path) and rospy.get_time() - self.navigation_start_time < self.config.MINIMUM_NAVIGATION_DURATION:
            final_goal_world = self.local_planner.mid_level_path[-1]
            current_pos_world = self._transform_pose_to_world(self.agent.get_state().position)
            dist_to_final_goal = np.linalg.norm((final_goal_world - current_pos_world)[:2])
            if dist_to_final_goal > 0.5:
                self._agent_turn(self.config.TURN_STEP_DEG * 2)
                return
        
        # If the path is finished, go back to scanning for a new frontier
        if self.current_target_index >= len(self.local_path):
            if self._execute_corner_recovery():
                return
            rospy.loginfo("Navigation to frontier complete. Transitioning to SCANNING.")
            navigation_duration_s = rospy.get_time() - self.navigation_start_time
            self.total_navigation_time += navigation_duration_s
            self.mid_planner.set_high_level_path(None)
            self.local_planner.set_mid_level_path(None)
            self.local_path = None
            self.current_target_index = -1
            self.state = "SCANNING"
            self.stage += 1
            # Save the executed path for this stage (we define path as the path that comes to this stage)
            next_stage_directory = os.path.join(self.config.BASE_DIRECTORY, str(self.stage))
            os.makedirs(next_stage_directory, exist_ok=True)
            path_filepath = os.path.join(next_stage_directory, "executed_path.json")
            path_data = {
                "habitat_frame_path": self.current_stage_path_habitat,
                "world_frame_path": self.current_stage_path_world
            }
            with open(path_filepath, "w") as json_file:
                json.dump(path_data, json_file)
            return

        target_pos = self.local_path[self.current_target_index]
        agent_state = self.agent.get_state()
        current_pos = agent_state.position
        delta_pos = target_pos - current_pos
        delta_pos[1] = 0
        distance = np.linalg.norm(delta_pos)

        if distance < self.config.DISTANCE_THRESHOLD_M:
            self.current_target_index += 1
            return

        target_angle = math.atan2(delta_pos[2], delta_pos[0])
        current_forward = quaternion.as_rotation_matrix(agent_state.rotation) @ np.array([0, 0, -1])
        current_angle = math.atan2(current_forward[2], current_forward[0])
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff <= -math.pi: angle_diff += 2 * math.pi

        if abs(angle_diff) > self.config.TURN_THRESHOLD_RAD:
            self._agent_turn(-np.sign(angle_diff) * self.config.TURN_STEP_DEG)
            self._record_agent_position()
        else:
            self._agent_forward(min(self.config.MOVE_STEP_M, distance))
            self._record_agent_position()

    
    def _archive_results(self):
        """Waits for scene graph files to be updated and stable, then copies them."""
        working_directory = os.path.join(self.config.BASE_DIRECTORY, str(self.stage))
        os.makedirs(working_directory, exist_ok=True)

        for filename in self.config.SCENE_GRAPH_FILENAMES:
            src_path = os.path.join(self.config.MAPPING_DIRECTORY, filename)
            
            rospy.loginfo(f"Waiting for updated and stable mapping file: {src_path}")
            previous_size = self.previous_dsg_sizes.get(src_path, -1)
            start_time = time.time()

            while time.time() - start_time < self.config.FILE_WAIT_TIMEOUT_S:
                if not os.path.exists(src_path):
                    time.sleep(1)
                    continue

                current_size = os.path.getsize(src_path)
                has_been_updated = (current_size != previous_size)
                is_large_enough = (current_size > self.config.MIN_DSG_FILE_SIZE_BYTES)

                if has_been_updated and is_large_enough:
                    # Stability check: wait to ensure the file is no longer being written to.
                    time.sleep(3)
                    size_after_wait = os.path.getsize(src_path)

                    if current_size == size_after_wait:
                        # File is stable, safe to copy.
                        dest_path = os.path.join(working_directory, filename)
                        shutil.copy(src_path, dest_path)
                        rospy.loginfo(f"Copied stable file to {dest_path}")
                        
                        # Update the size for the next cycle.
                        self.previous_dsg_sizes[src_path] = current_size
                        break
                
                time.sleep(2)
            else:
                rospy.logerr(f"Timeout! Did not find an updated/stable file at '{src_path}'.")


    def run_pipeline(self):
        while not rospy.is_shutdown() and not self.stop_requested:
            self.publish_tfs()
            if self.state in ["PLANNING", "NAVIGATING"]: self.publish_sensor_images()
            if self.state == "SCANNING": self.execute_scanning()
            elif self.state == "PLANNING": self.execute_planning()
            elif self.state == "NAVIGATING": self.execute_navigation()
            elif self.state == "FINISHED":
                rospy.loginfo_once("Exploration complete. No more frontiers found. Stopping.")
                time.sleep(1)
            self.rate.sleep()
        
        rospy.loginfo("="*40 + "\n      EXPLORATION SUMMARY\n" + "="*40)
        if self.exploration_start_time: rospy.loginfo(f"Total Duration: {time.time() - self.exploration_start_time:.2f} seconds")
        rospy.loginfo(f"Total Path Length: {self.total_path_length:.2f} meters")
        rospy.loginfo("="*40); self.sim.close(); rospy.loginfo("Pipeline shutdown.")

    def start_pipeline(self):
        if self.state == "IDLE":
            rospy.loginfo("User pressed 's'. Starting pipeline...")
            self.exploration_start_time = time.time()
            self.state = "SCANNING"
        else:
            rospy.logwarn(f"Cannot start, pipeline is already in '{self.state}' state.")

    def stop_pipeline(self):
        rospy.loginfo("Escape pressed. Requesting shutdown.")
        self.stop_requested = True

def setup_keyboard_listener(pipeline_instance):
    def on_press(key):
        try:
            if key.char == 's': 
                pipeline_instance.start_pipeline()
        except AttributeError:
            pass
    def on_release(key):
        if key == keyboard.Key.esc:
            pipeline_instance.stop_pipeline()
            return False
    print("\n" + "="*40 + "\n      COMPLETE FRONTIER EXPLORATION BASELINE      \n" + "="*40)
    print(" Press 's' to start the exploration cycle.\n Press 'Esc' to shut down.\n" + "="*40 + "\n")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

if __name__ == '__main__':
    try:
        with open("../config/frontier_config.yaml", "r") as f:
            raw_cfg = yaml.safe_load(f)
        camera_cfg = CameraConfig(**raw_cfg["CAMERA_CONFIG"])
        config = PipelineConfig(**{**raw_cfg, "CAMERA_CONFIG": camera_cfg})
        pipeline = FrontierPipeline(config)
        setup_keyboard_listener(pipeline)
        pipeline.run_pipeline()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred: {e}", exc_info=True)