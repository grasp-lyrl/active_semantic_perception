# This script sets up a Habitat simulator with keyboard controls to navigate an agent in a 3D environment.
import habitat_sim
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
import rosbag
import tf2_msgs
import geometry_msgs.msg
import tf2_ros
import math
import quaternion
from pynput import keyboard
from threading import Thread
from nav_msgs.msg import OccupancyGrid

move_cmd = {"forward": 0, "turn": 0, "strafe": 0}

def get_around_x_matrix(angle, rotation_only=False):
    # Axis is the X-axis: [1, 0, 0]
    if rotation_only:
        rot_q = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    else:
        rot_q = np.array([[1, 0, 0, 0],[0, np.cos(angle), -np.sin(angle), 0],[0, np.sin(angle), np.cos(angle), 0], [0, 0, 0, 1]])
    return rot_q

def get_around_y_matrix(angle, rotation_only=False):
    # Axis is the Y-axis: [0, 1, 0]
    if rotation_only:
        rot_q = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    else:
        rot_q = np.array([[np.cos(angle), 0, np.sin(angle), 0],[0, 1, 0, 0],[-np.sin(angle), 0, np.cos(angle), 0], [0, 0, 0, 1]])
    return rot_q

def get_around_z_matrix(angle, rotation_only=False):
    # Axis is the Z-axis: [0, 0, 1]
    if rotation_only:
        rot_q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        rot_q = np.array([[np.cos(angle), -np.sin(angle), 0, 0],[np.sin(angle), np.cos(angle), 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    return rot_q

def transform_habitat_pose_to_z_up(agent_s):
    """
    Transform Habitat agent_state pose (Y-up) into a Z-up coordinate system.
    """
    rot_q = get_around_x_matrix(-np.pi/2)
    rot_q_2 = get_around_z_matrix(np.pi/4)
    
    homogeneous_matrix = np.zeros((4, 4))
    homogeneous_matrix[:3, :3] = quaternion.as_rotation_matrix(agent_s.rotation)
    homogeneous_matrix[3, 3] = 1
    homogeneous_matrix[:3, 3] = agent_s.position
    new_matrix = homogeneous_matrix @ rot_q @ rot_q_2

    new_orientation = new_matrix[:3, :3]
    new_orientation = quaternion.from_rotation_matrix(new_orientation)
    new_position = new_matrix[:3, 3]
    
    return new_position, new_orientation

def transform_habitat_pose_to_z_front(agent_s):
    """
    Transform Habitat agent_state pose (Y-up) into a Z-front (Y-down) coordinate system.
    """
    rot_q = get_around_x_matrix(-np.pi)
    rot_q_2 = get_around_y_matrix(0)
    
    homogeneous_matrix = np.zeros((4, 4))
    homogeneous_matrix[:3, :3] = quaternion.as_rotation_matrix(agent_s.rotation)
    homogeneous_matrix[3, 3] = 1
    homogeneous_matrix[:3, 3] = agent_s.position
    new_matrix = homogeneous_matrix @ rot_q @ rot_q_2

    new_orientation = new_matrix[:3, :3]
    new_orientation = quaternion.from_rotation_matrix(new_orientation)
    new_position = new_matrix[:3, 3]
    
    return new_position, new_orientation


def create_transform_msg(transform_broadcaster, parent_frame, child_frame, agent_s, timestamp, mode):
    """Create a TransformStamped message."""
    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp = timestamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    
    if mode == 'z_up':
        position, orientation = transform_habitat_pose_to_z_up(agent_s)
    
    elif mode == 'z_front':
        position, orientation = transform_habitat_pose_to_z_front(agent_s)

    transform.transform.translation.x = position[0]
    transform.transform.translation.y = position[1]
    transform.transform.translation.z = position[2]

    transform.transform.rotation.w = orientation.w
    transform.transform.rotation.x = orientation.x
    transform.transform.rotation.y = orientation.y
    transform.transform.rotation.z = orientation.z
    transform_broadcaster.sendTransform(transform)

    return transform

def world_to_grid(current_grid, world_x, world_y):
    """Converts world coordinates to grid cell coordinates."""
    if not current_grid:
        return None

    origin_x = current_grid.info.origin.position.x
    origin_y = current_grid.info.origin.position.y
    resolution = current_grid.info.resolution

    grid_x = int(math.floor((world_x - origin_x) / resolution))
    grid_y = int(math.floor((world_y - origin_y) / resolution))

    return grid_x, grid_y

def grid_to_world(current_grid, grid_x, grid_y):
    """Converts grid cell coordinates to world coordinates (center of the cell)."""
    if not current_grid:
        return None

    origin_x = current_grid.info.origin.position.x
    origin_y = current_grid.info.origin.position.y
    resolution = current_grid.info.resolution

    world_x = (grid_x + 0.5) * resolution + origin_x
    world_y = (grid_y + 0.5) * resolution + origin_y

    return world_x, world_y


def add_patch_to_grid(grid_data_2d, map_info, patch_origin_x, patch_origin_y, patch_width_m, patch_height_m):
    """
    A helper function to add a single obstacle patch to the grid data.
    """
    OBSTACLE_VALUE = 100
    resolution = map_info.resolution
    map_origin = map_info.origin.position

    start_cell_x = int((patch_origin_x - map_origin.x) / resolution)
    start_cell_y = int((patch_origin_y - map_origin.y) / resolution)

    patch_width_cells = int(patch_width_m / resolution)
    patch_height_cells = int(patch_height_m / resolution)
    
    end_cell_x = start_cell_x + patch_width_cells
    end_cell_y = start_cell_y + patch_height_cells

    safe_start_y = max(0, start_cell_y)
    safe_end_y = min(map_info.height, end_cell_y)
    safe_start_x = max(0, start_cell_x)
    safe_end_x = min(map_info.width, end_cell_x)

    if safe_start_y < safe_end_y and safe_start_x < safe_end_x:
        grid_data_2d[safe_start_y:safe_end_y, safe_start_x:safe_end_x] = OBSTACLE_VALUE
        
    return grid_data_2d


def create_obstacle_patch(msg, publisher):
    grid_data_2d = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
    # Patch to scene 00573
    # grid_data_2d = add_patch_to_grid(
    #     grid_data_2d=grid_data_2d,
    #     map_info=msg.info,
    #     patch_origin_x=-6.3637237548828125 - 0.6,
    #     patch_origin_y=3.3085546493530273 - 0.5,
    #     patch_width_m=1.0,
    #     patch_height_m=0.4
    # )

    # grid_data_2d = add_patch_to_grid(
    #     grid_data_2d=grid_data_2d,
    #     map_info=msg.info,
    #     patch_origin_x=-3.034053726196289 - 2.0,
    #     patch_origin_y=-5.285573539733887 + 3.0,
    #     patch_width_m=1.0,
    #     patch_height_m=2.0
    # )

    # Patch to scene 00871
    # This is the patch for the closet
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=-2.0,
        patch_origin_y=8.0,
        patch_width_m=0.5,
        patch_height_m=2.0
    )

    # This is the patch for the bathroom
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=-4.25,
        patch_origin_y=6.3,
        patch_width_m=2.7,
        patch_height_m=0.4
    )

    # This is the additional patch for bedroom
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=-1.5,
        patch_origin_y=10.2,
        patch_width_m=3.5,
        patch_height_m=10
    )

    # This is the additional patch for bedroom
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=-0.5,
        patch_origin_y=-6.5,
        patch_width_m=0.3,
        patch_height_m=0.7
    )

    # This is the patch for the bedroom and living room
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=2.56239 - 0.2,
        patch_origin_y=3.82921 - 0.9,
        patch_width_m=10.0,
        patch_height_m=10.0
    )

    # This is the patch for the kitchen
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=2.4,
        patch_origin_y=0.2,
        patch_width_m=10.0,
        patch_height_m=3.0
    )

    # This is the patch for the laundry room
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=-5.8,
        patch_origin_y=-7.4,
        patch_width_m=4.0,
        patch_height_m=6.0
    )

    # This is the patch for the closet room
    grid_data_2d = add_patch_to_grid(
        grid_data_2d=grid_data_2d,
        map_info=msg.info,
        patch_origin_x=9.37,
        patch_origin_y=-7.8,
        patch_width_m=4.0,
        patch_height_m=4.0
    )

    modified_grid_msg = OccupancyGrid()
    modified_grid_msg.header = msg.header
    modified_grid_msg.header.stamp = rospy.Time.now()
    modified_grid_msg.info = msg.info
    modified_grid_msg.data = grid_data_2d.flatten().tolist()
    publisher.publish(modified_grid_msg)


def main():
    rospy.init_node('habitat_apartment_sim', anonymous=True)
    keyboard_thread = Thread(target=start_keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()


    rgb_pub = rospy.Publisher('/dominic/forward/color/image_raw', Image, queue_size=10)
    depth_pub = rospy.Publisher('/dominic/forward/depth/image_rect_raw', Image, queue_size=10)
    occupancy_modified_pub = rospy.Publisher('/occupancy_modified', OccupancyGrid, queue_size=1)
    occupancy_sub = rospy.Subscriber(
        '/clio_node/graph0/gvd/occupancy', 
        OccupancyGrid, 
        lambda msg: create_obstacle_patch(msg, occupancy_modified_pub)
    )
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    bridge = CvBridge()

    sim_cfg = habitat_sim.SimulatorConfiguration()
    # Apartment Scene (00362)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00362-o94q92w5PK5/o94q92w5PK5.basis.glb"
    # Office Scene (00111)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00111-AMEM2eWycTq/AMEM2eWycTq.basis.glb"
    # Apartment Scene (00871)
    sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/val/00871-VBzV5z6i1WS/VBzV5z6i1WS.basis.glb"
    # Apartment Scene (00853)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    # Apartment Scene (00712)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00712-HZ2iMMBsBQ9/HZ2iMMBsBQ9.basis.glb"
    # Apartment Scene (00701)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00701-tpxKD3awofe/tpxKD3awofe.basis.glb"
    # Apartment Scene (00573)
    # sim_cfg.scene_dataset_config_file = '/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/hm3d_annotated_basis.scene_dataset_config.json'
    # sim_cfg.scene_id = "/home/apple/Work/catkin_ws/habitat_datasets/hm3d_datasets/train/00573-1zDbEdygBeW/1zDbEdygBeW.basis.glb"
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]  # Height x Width
    rgb_sensor_spec.position = [0.0, 0.0, 0.0]  # Example: Camera position relative to agent
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.position = [0.0, 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    width, height = rgb_sensor_spec.resolution
    hfov = rgb_sensor_spec.hfov  # in degrees

    theta = math.radians(hfov)
    fx = width / (2.0 * math.tan(theta / 2.0))
    fy = height / (2.0 * math.tan(theta / 2.0))

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = [[fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1]]
    
    print("Intrinsic matrix: ", K)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    agent = sim.initialize_agent(0)
    agent_state = agent.get_state()
    agent_initial_state = agent.get_state()
    
    # Example: Set the agent in the middle of the apartment
    # Initial position for scene 00871
    agent_state.position = np.array([-5.5, 1.1, -3.0])
    # Initial position for scene 00853
    # agent_state.position = np.array([-0.5, 1.1, 4.5])
    # Initial position for scene 00573
    # agent_state.position = np.array([0.1, 1.1, 1.5])
    rotation_delta = habitat_sim.utils.common.quat_from_angle_axis(np.deg2rad(0), np.array([1, 0, 0]))
    agent_state.rotation = agent_state.rotation * rotation_delta
    agent.set_state(agent_state)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if move_cmd["forward"] != 0:
            agent_forward(move_cmd["forward"] * 0.075)
        if move_cmd["turn"] != 0:
            agent_turn(move_cmd["turn"] * 1.5)
        if move_cmd["strafe"] != 0:
            agent_strafe(move_cmd["strafe"] * 0.25)
        observations = sim.get_sensor_observations()

        rgb_image = cv2.cvtColor(observations['color_sensor'], cv2.COLOR_RGBA2RGB)
        depth_image = observations['depth_sensor']

        rgb_msg = bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
        depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')

        timestamp = rospy.Time.now()
        rgb_msg.header.stamp = timestamp
        depth_msg.header.stamp = timestamp

        rgb_pub.publish(rgb_msg)
        depth_pub.publish(depth_msg)

        agent_state = agent.get_state()
        create_transform_msg(tf_broadcaster, "world_habitat", "world", agent_initial_state, timestamp, mode='z_up')
        create_transform_msg(tf_broadcaster, "world_habitat", "dominic/forward_link", agent_state, timestamp, mode='z_front')

        rate.sleep()

        def agent_forward(dist):
            state = agent.get_state()
            forward = quaternion.as_rotation_matrix(sim.get_agent(0).get_state().rotation) @ np.array([0, 0, -1])
            forward[1] = 0
            state.position += forward * dist
            agent.set_state(state)

        def agent_turn(degrees):
            state = agent.get_state()
            rot = habitat_sim.utils.common.quat_from_angle_axis(math.radians(degrees), np.array([0, 1, 0]))
            state.rotation = rot * state.rotation
            agent.set_state(state)

        def agent_strafe(dist):
            state = agent.get_state()
            right = quaternion.as_rotation_matrix(sim.get_agent(0).get_state().rotation) @ np.array([1, 0, 0])
            right[1] = 0
            state.position += right * dist
            agent.set_state(state)


def on_press(key):
    global move_cmd
    try:
        if key.char == 'w':
            move_cmd["forward"] = 1
        elif key.char == 's':
            move_cmd["forward"] = -1
        elif key.char == 'a':
            move_cmd["turn"] = 1
        elif key.char == 'd':
            move_cmd["turn"] = -1
        elif key.char == 'q':
            move_cmd["strafe"] = -1
        elif key.char == 'e':
            move_cmd["strafe"] = 1
    except AttributeError:
        pass

def on_release(key):
    global move_cmd
    move_cmd = {"forward": 0, "turn": 0, "strafe": 0}
    if key == keyboard.Key.esc:
        return False  # Stop listener

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()


if __name__ == '__main__':
    main()
