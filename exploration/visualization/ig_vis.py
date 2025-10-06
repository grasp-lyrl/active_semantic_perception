import yaml
import numpy as np
import ast
import spark_dsg as dsg
import datetime
from scipy.spatial.transform import Rotation as R
import copy
import math
import datetime
from bidict import bidict
from dataclasses import dataclass

@dataclass(frozen=True)
class CameraConfig:
    fx: float = 320.0
    fy: float = 240.0
    cx: float = 319.5
    cy: float = 239.5
    width: int = 640
    height: int = 480
    near_clip: float = 0.5
    max_range: float = 3.0

class Camera:
    """
    Coordinate system -- Consistent with Clio
    (X-right, Y-down, Z-forward).
    """
    def __init__(self, config, position, yaw_degrees=0, world_up=np.array([0, 0, 1])):
        self.config = config
        self.position = position
        self.view_matrix, self.camera_quat = self._build_view_matrix(position, yaw_degrees, world_up)
        self.frustum_normals = self._calculate_frustum_normals()

    def _get_camera_orientation(self):
        return self.camera_quat

    def _build_view_matrix(self, pos, yaw_degrees, world_up):
        pitch_radians = -np.pi/18
        yaw_radians = np.radians(yaw_degrees)
        z_axis = np.array([
            np.cos(yaw_radians) * np.cos(pitch_radians),
            np.sin(yaw_radians) * np.cos(pitch_radians),
            np.sin(pitch_radians)
        ])
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, world_up) # Right = Forward x Up
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis) # Down = Forward x Right
        y_axis = y_axis / np.linalg.norm(y_axis)
    
        transformation = np.identity(4)
        transformation[0, :3] = x_axis
        transformation[1, :3] = y_axis
        transformation[2, :3] = z_axis

        t = transformation[:3, :3] @ (-pos)
        transformation[:3, 3] = t

        orientation_matrix = np.linalg.inv(transformation)[:3, :3]
        camera_quat = R.from_matrix(orientation_matrix).as_quat()
        
        return transformation, camera_quat

    def _calculate_frustum_normals(self):
        cfg = self.config
        scale_factor = cfg.fx / cfg.fy

        p_tl = np.array([-cfg.cx, -cfg.cy * scale_factor, cfg.fx])
        p_tr = np.array([cfg.width - cfg.cx, -cfg.cy * scale_factor, cfg.fx])
        p_br = np.array([cfg.width - cfg.cx, (cfg.height - cfg.cy) * scale_factor, cfg.fx])
        p_bl = np.array([-cfg.cx, (cfg.height - cfg.cy) * scale_factor, cfg.fx])

        normals = np.zeros((4, 3))
        normals[0] = np.cross(p_tr, p_tl)
        normals[1] = np.cross(p_br, p_tr)
        normals[2] = np.cross(p_bl, p_br)
        normals[3] = np.cross(p_tl, p_bl)

        for i in range(4):
            normals[i] /= np.linalg.norm(normals[i])
            
        return normals

    def point_is_in_view(self, point_world):
        point_world_h = np.append(point_world, 1)
        point_camera = (self.view_matrix @ point_world_h)[:3]

        if point_camera[2] < self.config.near_clip:
            return False

        if np.linalg.norm(point_camera) > self.config.max_range:
            return False

        for normal in self.frustum_normals:
            if np.dot(point_camera, normal) > 0:
                return False
        
        return True


def load_scene_graph(scene_graph_file):
    with open(scene_graph_file, 'r') as file:
        sample_data = yaml.safe_load(file)
    return sample_data

def smart_eval(expr):
    if isinstance(expr, str):
        return ast.literal_eval(expr)
    return expr

def calculate_bounding_box_corners(dimension, position, orientation_matrix):
    half_dim = dimension / 2.0

    corners_local = np.array([
        [-half_dim[0], -half_dim[1], -half_dim[2]],
        [+half_dim[0], -half_dim[1], -half_dim[2]],
        [+half_dim[0], +half_dim[1], -half_dim[2]],
        [-half_dim[0], +half_dim[1], -half_dim[2]],
        [-half_dim[0], -half_dim[1], +half_dim[2]],
        [+half_dim[0], -half_dim[1], +half_dim[2]],
        [+half_dim[0], +half_dim[1], +half_dim[2]],
        [-half_dim[0], +half_dim[1], +half_dim[2]],
    ])
    rotated_corners = np.dot(orientation_matrix, corners_local.T).T
    world_corners = rotated_corners + position

    return world_corners


def perturb_scene_graph(scene_graph, noise_low=-0.25, noise_high=0.25, seed=None):
    rng = np.random.default_rng(seed)
    sg = copy.deepcopy(scene_graph)

    for node in sg.get('nodes', []):
        if node.get('node_type') in ('object', 'structure'):
            pos = np.array(smart_eval(node['position']), dtype=float)
            pos += rng.uniform(noise_low, noise_high, size=3)
            node['position'] = str(list(pos))
    return sg

def build_scene_graph_groups(scene_graph_files, num_perturb=2, noise_low=-0.25, noise_high=0.25, seed=None):
    """
    Returns a list where each element corresponds to one original file and contains:
        [original_graph, perturb1, perturb2, ..., perturbN]
    """
    all_groups = []
    for path in scene_graph_files:
        original = load_scene_graph(path)
        group = [original]
        for k in range(num_perturb):
            perturb_seed = None if seed is None else seed + k
            group.append(perturb_scene_graph(original, noise_low, noise_high, perturb_seed))
        all_groups.append(group)
    return all_groups

def get_object_to_parent_mapping_for_one_graph(scene_graph):
    """
    Creates a mapping from an object ID to its parent's name.
    """
    nodes = scene_graph.get('nodes', [])
    edges = scene_graph.get('edges', [])

    node_id_to_name = {node['id']: node['name'] for node in nodes}
    node_id_to_type = {node['id']: node.get('node_type', 'unknown') for node in nodes}

    object_to_parent_name = {}
    for id1, id2 in edges:
        type1 = node_id_to_type[id1]
        type2 = node_id_to_type[id2]

        child_id, parent_id = None, None

        if type1 == 'object' and type2 == 'room':
            child_id, parent_id = id1, id2
        elif type2 == 'object' and type1 == 'room':
            child_id, parent_id = id2, id1
        else:
            continue

        if parent_id in node_id_to_name:
            parent_name = node_id_to_name[parent_id]
            object_to_parent_name[child_id] = parent_name
        else:
            print(f"Warning: Parent node with ID {parent_id} not found for child {child_id}.")

    return object_to_parent_name

def get_object_to_parent_mapping(filepaths):
    """
    Processes a list of scene graph YAML files to generate object-to-parent mappings.
    """
    all_mappings = []
    for filepath in filepaths:
        scene_graph = load_scene_graph(filepath)
        mapping = get_object_to_parent_mapping_for_one_graph(scene_graph)
        all_mappings.append(mapping)
    return all_mappings

def ray_intersects_obb(ray_origin, ray_direction, box_center, box_dims, box_orientation_matrix, id_tuple):
    """
    Checks if a ray intersects an Oriented Bounding Box (OBB) using the slab method.
    """
    inv_orientation = box_orientation_matrix.T
    ray_origin_local = inv_orientation @ (ray_origin - box_center)
    ray_direction_local = inv_orientation @ ray_direction
    half_dims = box_dims / 2.0
    min_bounds = -half_dims
    max_bounds = +half_dims
    
    t_min = 0.0
    t_max = np.inf

    for i in range(3):
        if abs(ray_direction_local[i]) < 1e-6:
            if ray_origin_local[i] < min_bounds[i] or ray_origin_local[i] > max_bounds[i]:
                return False, None
        else:
            t1 = (min_bounds[i] - ray_origin_local[i]) / ray_direction_local[i]
            t2 = (max_bounds[i] - ray_origin_local[i]) / ray_direction_local[i]
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            
            if t_min > t_max:
                return False, None

    if t_min <= t_max and t_max >= 0:
        intersection_distance = max(0.0, t_min)
        return True, intersection_distance
        
    return False, None

def visible_testing(scene_graph, position, yaw_angle):
    camera_config = CameraConfig()
    visible_objects_id = []
    visible_objects_name = {}
    camera = Camera(config=camera_config, position=position, yaw_degrees=yaw_angle)
    target_objects = []
    potential_occluders = []
    
    for node in scene_graph['nodes']:
        node_type = node['node_type']
        
        if node_type == 'object':
            orient_str = node.get('orientation')
            if orient_str is None:
                orient = np.eye(3)
            else:
                orient = np.array(smart_eval(orient_str), dtype=float)
            obj_data = {
                'id': node['id'],
                'name': node['name'],
                'pos': np.array(smart_eval(node['position'])),
                'dims': np.array(smart_eval(node['dimension'])),
                'orient': orient
            }
            target_objects.append(obj_data)
            
        elif node_type == 'structure':
            orient_str = node.get('orientation')
            if orient_str is None:
                orient = np.eye(3)
            else:
                orient = np.array(smart_eval(orient_str), dtype=float)
            occluder_data = {
                'id': node['id'],
                'pos': np.array(smart_eval(node['position'])),
                'dims': np.array(smart_eval(node['dimension'])),
                'orient': orient
            }
            potential_occluders.append(occluder_data)

    for target_obj in target_objects:
        corners = calculate_bounding_box_corners(target_obj['dims'], target_obj['pos'], target_obj['orient'])
        is_in_frustum = any(camera.point_is_in_view(corner) for corner in corners)
        
        if not is_in_frustum:
            continue

        is_occluded = False
        dist_to_target = np.linalg.norm(target_obj['pos'] - camera.position)
        
        if dist_to_target < 1e-6:
            visible_objects_id.append(target_obj['id'])
            continue

        ray_dir = (target_obj['pos'] - camera.position) / dist_to_target
        for occluder_obj in potential_occluders:
            if target_obj['id'] == occluder_obj['id']:
                continue

            intersects, intersection_dist = ray_intersects_obb(
                ray_origin=camera.position,
                ray_direction=ray_dir,
                box_center=occluder_obj['pos'],
                box_dims=occluder_obj['dims'],
                box_orientation_matrix=occluder_obj['orient'],
                id_tuple = (target_obj['id'], occluder_obj['id'])
            )

            # If it intersects and the occluder is closer than the target, it's occluded.
            if intersects and intersection_dist < dist_to_target:
                is_occluded = True
                break

        if not is_occluded:
            visible_objects_id.append(target_obj['id'])
            if target_obj['name'] not in visible_objects_name:
                visible_objects_name[target_obj['name']] = 1
            else:
                visible_objects_name[target_obj['name']] += 1
    camera_orientation = camera._get_camera_orientation()
    return visible_objects_id, camera.position, camera_orientation, visible_objects_name

def visualize_scene_graph(scene_graph, visible_objects, position, camera_orientation):
    start_object_node_id = 7998392938220000000
    start_room_node_id = 7782220156096220000
    mapping = bidict()
    G = dsg.DynamicSceneGraph([1,2,3,4])
    # This layer is for camera visualization
    G.create_dynamic_layer(dsg.DsgLayers.AGENTS,'a')
    for node in scene_graph['nodes']:
        attr = dsg._dsg_bindings.SemanticNodeAttributes()
        attr.name = node['name']
        attr.position = smart_eval(node['position'])
        node_type = node['node_type']
        if node_type == 'object' or node_type == 'structure':
            attr.bounding_box = dsg._dsg_bindings.BoundingBox()
            attr.bounding_box.world_P_center = smart_eval(node['position'])
            attr.bounding_box.dimensions = smart_eval(node['dimension'])
            try:
                attr.bounding_box.world_R_center = smart_eval(node['orientation'])
            except KeyError:
                attr.bounding_box.world_R_center = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            if node['id'] in visible_objects:
                attr.color = [255, 0, 0]
            else:
                attr.color = [0, 0, 0]
            G.add_node(dsg.DsgLayers.OBJECTS, start_object_node_id, attr)
            mapping[node['id']] = (start_object_node_id, 'object')
            start_object_node_id += 1

        
    for node in scene_graph['nodes']:
        room_visible = False
        attr = dsg._dsg_bindings.SemanticNodeAttributes()
        attr.name = node['name']
        attr.position = smart_eval(node['position'])
        node_type = node['node_type']
        if node_type == 'room':
            # Color visible rooms differently if they contain visible objects
            for edge in scene_graph['edges']:
                if edge[0] == node['id'] or edge[1] == node['id']:
                    other_id = edge[1] if edge[0] == node['id'] else edge[0]
                    if other_id in visible_objects and mapping[other_id][1] == 'object':
                        attr.color = [255, 0, 0]
                        room_visible = True
                        break

            if not room_visible:
                attr.color = [0, 0, 0]
            G.add_node(dsg.DsgLayers.ROOMS, start_room_node_id, attr)
            mapping[node['id']] = (start_room_node_id, 'room')
            start_room_node_id += 1

    for edge in scene_graph['edges']:
        id1, id2 = edge
        if id1 in mapping and id2 in mapping:
            if mapping[id1][1] == 'object' and mapping[id2][1] == 'object':
                continue
            attr = dsg._dsg_bindings.EdgeAttributes()
            attr.weight = 1.0
            attr.weighted = False
            G.insert_edge(mapping[id1][0], mapping[id2][0], attr)

    # Add the camera information
    camera_position = position
    quaternion = dsg._dsg_bindings.Quaterniond()
    quaternion.w = camera_orientation[3]
    quaternion.x = camera_orientation[0]
    quaternion.y = camera_orientation[1]
    quaternion.z = camera_orientation[2]
    camera_attr = dsg._dsg_bindings.AgentNodeAttributes()
    camera_attr.position = camera_position
    camera_attr.world_R_body = quaternion
    prefix = dsg._dsg_bindings.LayerPrefix('a')
    camera_time = datetime.timedelta()
    G.add_node(dsg.DsgLayers.AGENTS, prefix, camera_time, camera_attr)

    # Render the 3D scene graph
    dsg.render_to_open3d(G)

def main(scene_graph_file_list):
    for scene_graph_file in scene_graph_file_list:
        with open(scene_graph_file, 'r') as file:
            scene_graph = yaml.safe_load(file)
        min_position = None
        max_position = None
        positions = [smart_eval(node['position']) for node in scene_graph['nodes']]
        positions_array = np.array(positions)
        file_min = positions_array.min(axis=0)
        file_max = positions_array.max(axis=0)
        if min_position is None:
            min_position = file_min
            max_position = file_max
        else:
            min_position = np.minimum(min_position, file_min)
            max_position = np.maximum(max_position, file_max)

        # Quadrotor can only be in fixed height
        min_position[2] = 1.0
        max_position[2] = 1.0
        low_bounds = np.append(min_position, 0.0)
        high_bounds = np.append(max_position, 360.0)
        accepted_samples = 0
        num_samples = 1
        while accepted_samples < num_samples:
            sample = np.array([0.81700292, -0.54746275, 1.0, 230.2])
            print(f"Sample: {sample}")
            pos = sample[:3]
            yaw_deg = float(sample[3])
            group_observations = []
            any_visible = False
            # All the scene graphs we built (M*N, M is number of scene graph in pipeline, N is number of scene graph we asked LLM to predict)
            ids, cam_pos, cam_quat, name_counts = visible_testing(scene_graph, pos, yaw_deg)
            if ids:
                any_visible = True
            if any_visible:
                # Consider a valid sample if it has visible objects
                accepted_samples += 1
        visualize_scene_graph(scene_graph, ids, pos, cam_quat)


def calculate_uncertainty(scene_graph_files, num_samples, num_perturb, noise_low, noise_high, weight, seed=None):
    rng = np.random.default_rng(seed)
    # Construct groups for existing scene graph (each group contains original graph + perturbed graph)
    groups = build_scene_graph_groups(scene_graph_files, num_perturb=num_perturb, noise_low=noise_low, noise_high=noise_high, seed=seed)
    mapping = get_object_to_parent_mapping(scene_graph_files)

    min_position = None
    max_position = None
    scene_graphs = []
    for group in groups:
        for scene_graph in group:
            scene_graphs.append(scene_graph)
            positions = [smart_eval(node['position']) for node in scene_graph['nodes']]
            positions_array = np.array(positions)
            file_min = positions_array.min(axis=0)
            file_max = positions_array.max(axis=0)
            if min_position is None:
                min_position = file_min
                max_position = file_max
            else:
                min_position = np.minimum(min_position, file_min)
                max_position = np.maximum(max_position, file_max)

    # Quadrotor can only be in fixed height
    min_position[2] = 1.0
    max_position[2] = 1.0
    print(f"Min position: {min_position}, Max position: {max_position}")

    accepted_samples = 0

    low_bounds = np.append(min_position, 0.0)
    high_bounds = np.append(max_position, 360.0)


    # Sample num_samples times
    sample_observations = []
    while accepted_samples < num_samples:
        # Consider the sample acceptable if it has visible objects
        sample = rng.uniform(low=low_bounds, high=high_bounds)
        pos = sample[:3]
        yaw_deg = float(sample[3])
        group_observations = []
        any_visible = False
        # All the scene graphs we built (M*N, M is number of scene graph in pipeline, N is number of scene graph we asked LLM to predict)
        for i, group in enumerate(groups):
            visible_ids_all = []
            names_list = []  # To store counts of visible objects (for example [{chair: 2}, {table: 1}])
            rooms_list = []  # To store sets of visible rooms (for example [{living room}, {kitchen, living room}])
            # Scene graphs in a group contain original plus perturbed versions
            for sg in group:
                ids, cam_pos, cam_quat, name_counts = visible_testing(sg, pos, yaw_deg)
                if ids:
                    any_visible = True
                    visible_ids_all.extend(ids)

                # Determine the single room for this viewpoint by majority vote
                final_room_observation = set()
                if ids: # Only proceed if at least one object is visible
                    room_object_counts = {}
                    # 1. Count the number of visible objects per room
                    for obj_id in ids:
                        if obj_id in mapping[i]:
                            parent_name = mapping[i][obj_id]
                            room_object_counts[parent_name] = room_object_counts.get(parent_name, 0) + 1
                    
                    # 2. Find the room with the highest object count
                    if room_object_counts:
                        majority_vote_room = max(room_object_counts, key=room_object_counts.get)
                        final_room_observation = {majority_vote_room}

                rooms_list.append(final_room_observation)
                names_list.append(name_counts)
            group_observations.append((sample, visible_ids_all, cam_pos, cam_quat, names_list, rooms_list, yaw_deg))
        if any_visible:
            # Consider a valid sample if it has visible objects
            sample_observations.append(group_observations)
            accepted_samples += 1

    def _calculate_ig_components(group_obs, list_extractor):
        """Helper to calculate entropy components (H_y and H_y_epsilon)."""
        total_counts = {}
        group_entropies = []
        
        for obs_tuple in group_obs:
            counts_list = list_extractor(obs_tuple)
            group_counts = {}
            for d in counts_list:
                obs = frozenset(d.items() if isinstance(d, dict) else d)
                group_counts[obs] = group_counts.get(obs, 0) + 1
                total_counts[obs] = total_counts.get(obs, 0) + 1
            
            gtot = sum(group_counts.values())
            H_g = -sum((c / gtot) * math.log2(c / gtot) for c in group_counts.values()) if gtot > 0 else 0.0
            group_entropies.append(H_g)

        H_y_epsilon = float(sum(group_entropies) / max(len(group_entropies), 1))
        tot = sum(total_counts.values())
        H_y = -sum((c / tot) * math.log2(c / tot) for c in total_counts.values()) if tot > 0 else 0.0
        return {"H_y_epsilon": H_y_epsilon, "H_y": H_y}
    
    object_uncertainty_list = []
    room_uncertainty_list = []

    for i, group_obs in enumerate(sample_observations):
        _pos = group_obs[0][2]
        _yaw = group_obs[0][-1]

        # Calculate for OBJECTS
        obj_uncertainty = _calculate_ig_components(group_obs, lambda obs: obs[4])
        object_uncertainty_list.append(obj_uncertainty)
        print(f"Sample {i + 1} (Objects) @ {_pos}, yaw {_yaw:.1f}: H_y_epsilon={obj_uncertainty['H_y_epsilon']:.4f}, H_y={obj_uncertainty['H_y']:.4f}")

        # Calculate for ROOMS
        room_uncertainty = _calculate_ig_components(group_obs, lambda obs: obs[5])
        room_uncertainty_list.append(room_uncertainty)
        print(f"Sample {i + 1} (Rooms)   @ {_pos}, yaw {_yaw:.1f}: H_y_epsilon={room_uncertainty['H_y_epsilon']:.4f}, H_y={room_uncertainty['H_y']:.4f}")

    best_ig = -1.0
    best_obj_ig = -1.0
    best_room_ig = -1.0
    best_index_list = []
    for i, (obj_u, room_u) in enumerate(zip(object_uncertainty_list, room_uncertainty_list)):
        obj_ig = obj_u["H_y"] - obj_u["H_y_epsilon"]
        room_ig = room_u["H_y"] - room_u["H_y_epsilon"]
        total_ig = obj_ig + weight * room_ig
        if total_ig - best_ig > 1e-6:
            best_ig = total_ig
            best_obj_ig = obj_ig
            best_room_ig = room_ig
            best_index_list = [i + 1]
        elif abs(total_ig - best_ig) <= 1e-6:
            best_index_list.append(i + 1)

    print(f"Highest Information Gain: {best_ig}, Object Information Gain: {best_obj_ig}, Room Information Gain: {best_room_ig}")
    print(f"Highest Sample: {best_index_list}")


if __name__ == "__main__":
    scene_graph_files = [
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph0.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph1.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph2.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph3.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph4.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph5.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph6.yaml",
        "/home/apple/Work/scene_graph_processing/Summer_research/Previous_Experiments/Scene_Graph_Ensemble_2/habitat_scene_graph_new_graph7.yaml"
    ]
    # calculate_uncertainty(scene_graph_files, num_samples=300, num_perturb=3, noise_low=-0.25, noise_high=0.25, weight=1, seed=42)
    main(scene_graph_files)