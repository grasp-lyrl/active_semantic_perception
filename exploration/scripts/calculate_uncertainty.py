import yaml
import numpy as np
import ast
import datetime
from scipy.spatial.transform import Rotation as R
import copy
import math
from dataclasses import dataclass
import json
import datetime
import os

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
    
class UncertaintyCalculator:
    """
    Calculates the most informative next viewpoint to explore based on a set of
    latent complete scene graphs.
    """
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(self.config.SEED)
        self.camera_config = config.CAMERA_CONFIG

    def select_next_target(self, scene_graph_files, history_pos):
        """
        Main entry point. Calculates uncertainty over a set of scene graphs and
        returns the viewpoint with the highest information gain.
        """
        print("Starting uncertainty calculation to select next target...")

        # 1. Build scene graph groups (original + perturbations) and mapping for each file
        groups = self._build_scene_graph_groups(scene_graph_files)
        mapping = self._get_object_to_parent_mapping(scene_graph_files)

        # 2. Determine the sampling bounds from all graph nodes
        min_pos, max_pos = self._get_sampling_bounds(groups)

        # 3. Sample potential viewpoints and calculate observations for each
        sample_observations = self._gather_sample_observations(groups, mapping, min_pos, max_pos)

        # 4. Calculate information gain for each sample and find the best one
        best_sample_pose_list, best_ig = self._find_best_viewpoint(sample_observations)

        best_filtered_list = []

        for pose in best_sample_pose_list:
            is_far_enough = all(
                np.linalg.norm(pose[:3] - hist_pos) > self.config.MINIMUM_DISTANCE_DIFFERENCE
                for hist_pos in history_pos
            )

            if is_far_enough:
                best_filtered_list.append(pose)

        # Prepare the data for JSON serialization (convert numpy arrays to lists)
        data_to_save = {
            "timestamp": datetime.datetime.now().isoformat(),
            "best_sample_poses": [pose.tolist() for pose in best_sample_pose_list],
            "best_filtered_poses": [pose.tolist() for pose in best_filtered_list],
            "best_information_gain": float(best_ig)
        }

        json_filepath = os.path.join(self.config.WORKING_DIRECTORY, "pose_log.json")
        # Append the data as a new line in a JSON file
        with open(json_filepath, "a") as f:
            f.write(json.dumps(data_to_save) + "\n")

        # Rarely useful. The information gain metric usually yields only one viewpoint, or viewpoints that are very close to each other.
        if best_filtered_list:
            # print(f"Found {len(best_filtered_list)} valid viewpoints that meet the distance criteria. They are {best_filtered_list}")
            return self.rng.choice(best_filtered_list)
        else:
            # print("No valid viewpoints found that meet the distance criteria. Selecting random viewpoint instead.")
            return self.rng.choice(best_sample_pose_list)

    def _get_object_to_parent_mapping_for_one_graph(self, scene_graph):
        """
        Creates a mapping from an object ID to its parent's name.
        """
        nodes = scene_graph.get('nodes', [])
        edges = scene_graph.get('edges', [])

        # 1. Create a lookup dictionary for node IDs to names for efficient access.
        # This avoids repeatedly searching the nodes list.
        node_id_to_name = {node['id']: node['name'] for node in nodes}
        node_id_to_type = {node['id']: node.get('node_type', 'unknown') for node in nodes}

        # 2. Create the mapping from object ID to the parent's name.
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

            # Ensure the parent ID exists in our name lookup map before access.
            if parent_id in node_id_to_name:
                parent_name = node_id_to_name[parent_id]
                object_to_parent_name[child_id] = parent_name
            else:
                # Not likely to happen
                print(f"Warning: Parent node with ID {parent_id} not found for child {child_id}.")

        return object_to_parent_name

    def _get_object_to_parent_mapping(self, filepaths):
        """
        Processes a list of scene graph YAML files to generate object-to-parent mappings.
        """
        all_mappings = []
        for filepath in filepaths:
            scene_graph = self._load_scene_graph(filepath)
            mapping = self._get_object_to_parent_mapping_for_one_graph(scene_graph)
            all_mappings.append(mapping)
        return all_mappings


    def _load_scene_graph(self, scene_graph_file):
        with open(scene_graph_file, 'r') as file:
            return yaml.safe_load(file)

    def _smart_eval(self, expr):
        return ast.literal_eval(expr) if isinstance(expr, str) else expr

    def _perturb_scene_graph(self, scene_graph, seed):
        sg = copy.deepcopy(scene_graph)
        noise_low = self.config.UNCERTAINTY_NOISE_LOW
        noise_high = self.config.UNCERTAINTY_NOISE_HIGH
        for node in sg.get('nodes', []):
            if node.get('node_type') in ('object', 'structure'):
                pos = np.array(self._smart_eval(node['position']), dtype=float)
                pos += self.rng.uniform(noise_low, noise_high, size=3)
                node['position'] = str(list(pos))
        return sg

    def _build_scene_graph_groups(self, scene_graph_files):
        all_groups = []
        for path in scene_graph_files:
            original = self._load_scene_graph(path)
            group = [original]
            for k in range(self.config.UNCERTAINTY_PERTURBATIONS):
                group.append(self._perturb_scene_graph(original, seed=self.config.SEED + k))
            all_groups.append(group)
        return all_groups

    def _get_sampling_bounds(self, groups):
        min_position, max_position = None, None
        for group in groups:
            for scene_graph in group:
                positions = [self._smart_eval(n['position']) for n in scene_graph['nodes']]
                positions_array = np.array(positions)
                file_min = positions_array.min(axis=0)
                file_max = positions_array.max(axis=0)
                if min_position is None:
                    min_position, max_position = file_min, file_max
                else:
                    min_position = np.minimum(min_position, file_min)
                    max_position = np.maximum(max_position, file_max)
        
        # Clamp Z-axis for agent height
        min_position[2] = self.config.AGENT_HEIGHT
        max_position[2] = self.config.AGENT_HEIGHT
        
        return min_position, max_position

    def _gather_sample_observations(self, groups, mapping, min_position, max_position):
        low_bounds = np.append(min_position, 0.0)
        high_bounds = np.append(max_position, 360.0)
        
        sample_observations = []
        accepted_samples = 0

        print(f"Sampling {self.config.UNCERTAINTY_SAMPLES} viewpoints within bounds: {low_bounds} to {high_bounds}")
        while accepted_samples < self.config.UNCERTAINTY_SAMPLES:
            sample = self.rng.uniform(low=low_bounds, high=high_bounds)
            pos, yaw_deg = sample[:3], float(sample[3])
            
            group_obs_for_sample = []
            any_visible = False
            for i, group in enumerate(groups):
                names_list = []
                rooms_list = []
                for sg in group:
                    ids, _, _, name_counts = self._visible_testing(sg, pos, yaw_deg)
                    if name_counts:
                        any_visible = True
                    # Determine the room by majority vote of visible objects' parents
                    final_room_observation = set()
                    if ids:
                        room_counts = {}
                        for obj_id in ids:
                            if obj_id in mapping[i]:
                                parent_name = mapping[i][obj_id]
                                room_counts[parent_name] = room_counts.get(parent_name, 0) + 1

                        if room_counts:
                            majority_vote_room = max(room_counts, key=room_counts.get)
                            final_room_observation = {majority_vote_room}
                    names_list.append(name_counts)
                    rooms_list.append(final_room_observation)

                group_obs_for_sample.append({
                    'sample': sample,
                    'names_list': names_list,
                    'rooms_list': rooms_list
                })

            if any_visible:
                sample_observations.append(group_obs_for_sample)
                accepted_samples += 1

        return sample_observations
    
    def _calculate_ig_components(self, group_obs_list, list_extractor):
        total_counts = {}
        group_entropies = []

        for group_obs in group_obs_list:
            # Extract the list of observations (e.g., names_list) for one group
            counts_list = list_extractor(group_obs)
            group_counts = {}
            for item in counts_list:
                obs = frozenset(item.items() if isinstance(item, dict) else item)
                group_counts[obs] = group_counts.get(obs, 0) + 1
                total_counts[obs] = total_counts.get(obs, 0) + 1

            gtot = sum(group_counts.values())
            H_g = -sum((c / gtot) * math.log2(c / gtot) for c in group_counts.values()) if gtot > 0 else 0.0
            group_entropies.append(H_g)

        H_y_epsilon = float(sum(group_entropies) / max(len(group_entropies), 1))
        tot = sum(total_counts.values())
        H_y = -sum((c / tot) * math.log2(c / tot) for c in total_counts.values()) if tot > 0 else 0.0

        return {"H_y_epsilon": H_y_epsilon, "H_y": H_y}

    def _find_best_viewpoint(self, sample_observations):
        """
        Calculates the combined information gain for each sample and returns the pose
        with the highest gain.
        """
        best_ig = -1.0
        best_pose_list = []

        for i, group_obs_list in enumerate(sample_observations):
            # Calculate IG for OBJECTS
            obj_uncertainty = self._calculate_ig_components(group_obs_list, lambda obs: obs['names_list'])
            obj_ig = obj_uncertainty["H_y"] - obj_uncertainty["H_y_epsilon"]

            # Calculate IG for ROOMS
            room_uncertainty = self._calculate_ig_components(group_obs_list, lambda obs: obs['rooms_list'])
            room_ig = room_uncertainty["H_y"] - room_uncertainty["H_y_epsilon"]

            # Calculate the final weighted Information Gain
            total_ig = obj_ig + self.config.UNCERTAINTY_ROOM_WEIGHT * room_ig

            # Check if this sample is better than the current best
            sample_pose = group_obs_list[0]['sample']
            if total_ig - best_ig > 1e-6: # New best found
                best_ig = total_ig
                best_pose_list = [sample_pose]
            elif abs(total_ig - best_ig) <= 1e-6:
                best_pose_list.append(sample_pose)

        print(f"Highest Combined Information Gain found: {best_ig:.4f}")
        return best_pose_list, best_ig

    def _calculate_bounding_box_corners(self, dimension, position, orientation_matrix):
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
    
    def _ray_intersects_obb(self, ray_origin, ray_direction, box_center, box_dims, box_orientation_matrix):
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

    def _visible_testing(self, scene_graph, position, yaw_angle):
        visible_objects_id = []
        visible_objects_name = {}
        camera = Camera(config=self.camera_config, position=position, yaw_degrees=yaw_angle)
        target_objects = []
        potential_occluders = []
        
        for node in scene_graph['nodes']:
            node_type = node['node_type']
            
            # We only care about the visibility of 'objects'.
            if node_type == 'object':
                orient_str = node.get('orientation')
                if orient_str is None:
                    orient = np.eye(3)
                else:
                    orient = np.array(self._smart_eval(orient_str), dtype=float)
                obj_data = {
                    'id': node['id'],
                    'name': node['name'],
                    'pos': np.array(self._smart_eval(node['position'])),
                    'dims': np.array(self._smart_eval(node['dimension'])),
                    'orient': orient
                }
                target_objects.append(obj_data)
                
            # 'Structures' can only occlude, they are not targets themselves.
            elif node_type == 'structure':
                orient_str = node.get('orientation')
                if orient_str is None:
                    orient = np.eye(3)
                else:
                    orient = np.array(self._smart_eval(orient_str), dtype=float)
                occluder_data = {
                    'id': node['id'],
                    'pos': np.array(self._smart_eval(node['position'])),
                    'dims': np.array(self._smart_eval(node['dimension'])),
                    'orient': orient
                }
                potential_occluders.append(occluder_data)

        for target_obj in target_objects:
            corners = self._calculate_bounding_box_corners(target_obj['dims'], target_obj['pos'], target_obj['orient'])
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

                # Check if the ray to the target intersects the occluder's bounding box.
                intersects, intersection_dist = self._ray_intersects_obb(
                    ray_origin=camera.position,
                    ray_direction=ray_dir,
                    box_center=occluder_obj['pos'],
                    box_dims=occluder_obj['dims'],
                    box_orientation_matrix=occluder_obj['orient']
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