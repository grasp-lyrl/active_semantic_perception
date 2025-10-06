#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import math
import heapq
import collections
from tf2_msgs.msg import TFMessage
import message_filters
import geometry_msgs.msg
import numpy as np
import ast
from scipy.ndimage import binary_dilation


# MidPlanner refers to planning using the global occupancy grid. The logic is very similar to local_planning.py.
class MidPlanner:
    def __init__(self, config, tf_buffer):
        self.config = config
        self.tf_buffer = tf_buffer
        self.gvd_map_pub = rospy.Publisher('/gvd_merged_map', OccupancyGrid, queue_size=1)

        # State variables
        self.high_level_path = None
        self.current_grid = None
        self.planning_active = False

        self.path_pub = rospy.Publisher(self.config.MID_PATH_TOPIC, Path, queue_size=1)

        grid_topics = [
            f"{self.config.OCCUPANCY_GRID_TOPIC_PREFIX}{i}/gvd/occupancy"
            for i in range(self.config.OCCUPANCY_GRAPH_COUNT)
        ]
        self.grid_subs = [message_filters.Subscriber(topic, OccupancyGrid) for topic in grid_topics]
        self.grid_sync = message_filters.ApproximateTimeSynchronizer(self.grid_subs, 10, 0.1)
        self.grid_sync.registerCallback(self._grid_callback)

    def set_high_level_path(self, path):
        self.high_level_path = path

    def _grid_callback(self, *grids):
        # Only plan if we have a job to do and are not already busy
        # Fix for frontier baseline
        self.current_grid = self.merge_occupancy_grids(grids)

        if not self.high_level_path or self.planning_active:
            return

        try:
            self.planning_active = True
            self.plan_path()
        finally:
            self.planning_active = False

    def plan_path(self):
        # Determine the furthest reachable point along the high-level path
        mid_target_pos = self.high_level_path[-1]

        if not self.current_grid:
            rospy.logwarn("Mid planner cannot plan: missing grid")
            self.publish_empty_path()
            return

        # Get the agent's current pose
        try:
            transform = self.tf_buffer.lookup_transform(
                self.current_grid.header.frame_id,
                self.config.AGENT_FRAME,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            start_pos_world = transform.transform.translation
        except Exception as e:
            rospy.logerr(f"TF Error getting agent pose: {e}")
            self.publish_empty_path()
            return

        # Convert world coordinates to grid cells
        start_cell = self.world_to_grid(start_pos_world.x, start_pos_world.y)
        goal_cell = self.world_to_grid(mid_target_pos[0], mid_target_pos[1])

        potential_targets = self.find_reachable_goal_near(mid_target_pos[0], mid_target_pos[1], top_k=10)

        if potential_targets is None:
            potential_targets = []

        if self.is_valid_cell(goal_cell[0], goal_cell[1], navigation=True):
            heapq.heappush(potential_targets, (0, goal_cell))

        # Run A* search
        grid_path = None
        while not grid_path and potential_targets:
            target_cell = heapq.heappop(potential_targets)[1]
            grid_path, path_length_world, unknown_count = self.a_star_search(start_cell, target_cell)

        if grid_path:
            self.publish_path(grid_path)
        else:
            rospy.logwarn("A* search failed to find a mid path.")
            self.publish_empty_path()

    def add_patch_to_grid(self, grid_data_2d, map_origin, map_resolution, map_height, map_width, 
                      patch_origin_x, patch_origin_y, patch_width_m, patch_height_m, patch_value=100):
        # Convert world coordinates to grid cell indices
        start_cell_x = int((patch_origin_x - map_origin.position.x) / map_resolution)
        start_cell_y = int((patch_origin_y - map_origin.position.y) / map_resolution)

        # Calculate patch size in grid cells
        patch_width_cells = int(patch_width_m / map_resolution)
        patch_height_cells = int(patch_height_m / map_resolution)
        end_cell_x = start_cell_x + patch_width_cells
        end_cell_y = start_cell_y + patch_height_cells

        # Clamp coordinates to be safely within map boundaries
        safe_start_y = max(0, start_cell_y)
        safe_end_y = min(map_height, end_cell_y)
        safe_start_x = max(0, start_cell_x)
        safe_end_x = min(map_width, end_cell_x)

        # Modify the grid using NumPy slicing if the area is valid
        if safe_start_y < safe_end_y and safe_start_x < safe_end_x:
            grid_data_2d[safe_start_y:safe_end_y, safe_start_x:safe_end_x] = patch_value
            
        return grid_data_2d

    def merge_occupancy_grids(self, grid_msgs):
        reference_grid = grid_msgs[0]
        ref_resolution = reference_grid.info.resolution
        ref_orientation = reference_grid.info.origin.orientation

        # --- 1. Determine the Bounds of the New Combined Grid ---
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for grid in grid_msgs:
            origin_x = grid.info.origin.position.x
            origin_y = grid.info.origin.position.y
            
            min_x = min(min_x, origin_x)
            min_y = min(min_y, origin_y)

            max_x = max(max_x, origin_x + grid.info.width * ref_resolution)
            max_y = max(max_y, origin_y + grid.info.height * ref_resolution)

        # --- 2. Define the New Grid's Properties ---
        new_origin = geometry_msgs.msg.Pose()
        new_origin.position.x = min_x
        new_origin.position.y = min_y
        new_origin.orientation = ref_orientation

        new_width = int(math.ceil((max_x - min_x) / ref_resolution))
        new_height = int(math.ceil((max_y - min_y) / ref_resolution))

        merged_grid_data_2d = np.full((new_height, new_width), -1, dtype=np.int8)

        # --- 3. Copy Data from Each Source Grid to the New Grid ---
        for grid in grid_msgs:
            source_grid_2d = np.array(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width)

            x_offset = int(round((grid.info.origin.position.x - new_origin.position.x) / ref_resolution))
            y_offset = int(round((grid.info.origin.position.y - new_origin.position.y) / ref_resolution))

            h, w = grid.info.height, grid.info.width

            target_region = merged_grid_data_2d[y_offset:y_offset + h, x_offset:x_offset + w]
            
            merged_grid_data_2d[y_offset:y_offset + h, x_offset:x_offset + w] = np.maximum(
                target_region, source_grid_2d
            )
        # Patchify the occupancy grid to fix mesh errors (Specific to habitat scene 00573)
        if self.config.SCENE_NUMBER == 573:
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-6.9637237548828125,
                patch_origin_y=2.8085546493530273,
                patch_width_m=1.0,
                patch_height_m=0.4,
                patch_value=100
            )

            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-5.034053726196289,
                patch_origin_y=-2.285573539733887,
                patch_width_m=1.0,
                patch_height_m=2.0,
                patch_value=100
            )

        # Patchify the occupancy grid to fix mesh errors (Specific to habitat scene 00871)
        if self.config.SCENE_NUMBER == 871:
            # This is the patch for the bedroom
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-1.5,
                patch_origin_y=10.2,
                patch_width_m=3.5,
                patch_height_m=10,
                patch_value=100
            )
            
            # This is the patch for the bathroom
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-4.25,
                patch_origin_y=6.3,
                patch_width_m=2.7,
                patch_height_m=0.4,
                patch_value=100
            )

            # This is the additional patch for the laundry room
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                    map_resolution=ref_resolution,
                    map_height=new_height,
                    map_width=new_width,
                patch_origin_x=-0.5,
                patch_origin_y=-6.5,
                patch_width_m=0.3,
                patch_height_m=0.7,
                patch_value=100
            )

            # This is the patch for the closet
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-2.0,
                patch_origin_y=8.0,
                patch_width_m=0.5,
                patch_height_m=2.0,
                patch_value=100
            )

            # This is the patch for bedroom and living room
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=2.5623907895073317 - 0.2,
                patch_origin_y=3.8292140821823435 - 0.9,
                patch_width_m=10.0,
                patch_height_m=10.0,
                patch_value=100
            )
            # This is the patch for the kitchen
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=2.4,
                patch_origin_y=0.2,
                patch_width_m=10.0,
                patch_height_m=3.0,
                patch_value=100
            )
            # This is the patch for the laundry room
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=-5.8,
                patch_origin_y=-7.4,
                patch_width_m=4.0,
                patch_height_m=6.0,
                patch_value=100
            )
            # This is the patch for the closet room
            merged_grid_data_2d = self.add_patch_to_grid(
                grid_data_2d=merged_grid_data_2d,
                map_origin=new_origin,
                map_resolution=ref_resolution,
                map_height=new_height,
                map_width=new_width,
                patch_origin_x=9.37,
                patch_origin_y=-7.8,
                patch_width_m=4.0,
                patch_height_m=4.0,
                patch_value=100
            )

        # --- 4. Finalize and Return the New Grid Message ---
        merged_grid_msg = OccupancyGrid()
        merged_grid_msg.header = reference_grid.header # Use header from the first grid
        merged_grid_msg.info.resolution = ref_resolution
        merged_grid_msg.info.width = new_width
        merged_grid_msg.info.height = new_height
        merged_grid_msg.info.origin = new_origin

        merged_grid_msg.data = merged_grid_data_2d.flatten().tolist()

        self.gvd_map_pub.publish(merged_grid_msg)

        return merged_grid_msg


    def find_mid_planner_target(self):
        info = self.current_grid.info

        for i in range(len(self.high_level_path)     - 1, 0, -1):
            p_start, p_end = self.high_level_path[i - 1], self.high_level_path[i]
            segment_len = math.dist(p_start, p_end)
            
            num_steps = int(math.ceil(segment_len / (info.resolution / 2.0)))
                
            for step in range(num_steps, -1, -1):
                alpha = step / num_steps
                sampled_point = (
                    p_start[0] * (1 - alpha) + p_end[0] * alpha,
                    p_start[1] * (1 - alpha) + p_end[1] * alpha
                )
                
                grid_coords = self.world_to_grid(sampled_point[0], sampled_point[1])

                if grid_coords:
                    grid_x, grid_y = grid_coords
                    
                    # Check bounds
                    if 0 <= grid_y < info.height and 0 <= grid_x < info.width:
                        index = grid_y * info.width + grid_x
                        if self.current_grid.data[index] == 0:
                            return sampled_point
                            
        return None

    def world_to_grid(self, world_x, world_y):
        if not self.current_grid:
            return None

        origin_x = self.current_grid.info.origin.position.x
        origin_y = self.current_grid.info.origin.position.y
        resolution = self.current_grid.info.resolution

        grid_x = int(math.floor((world_x - origin_x) / resolution))
        grid_y = int(math.floor((world_y - origin_y) / resolution))

        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        if not self.current_grid:
            return None

        origin_x = self.current_grid.info.origin.position.x
        origin_y = self.current_grid.info.origin.position.y
        resolution = self.current_grid.info.resolution

        world_x = (grid_x + 0.5) * resolution + origin_x
        world_y = (grid_y + 0.5) * resolution + origin_y

        return world_x, world_y

    def is_valid_cell(self, grid_x, grid_y, navigation=False):
        if not self.current_grid:
            return False

        width = self.current_grid.info.width
        height = self.current_grid.info.height

        # Check bounds
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return False

        index = grid_y * width + grid_x

        state = self.current_grid.data[index]
        if state == -1:
            if navigation:
                return False # Treat unknown as not traversable for navigation
            else:
                return True # Treat unknown as traversable
        elif state == 100:
            return False # Obstacle
        elif state  == 0:
            return True # Free space
        else:
            return False # Invalid state

    def get_neighbors(self, grid_x, grid_y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue # Skip self
                
                nx, ny = grid_x + dx, grid_y + dy
                if self.is_valid_cell(nx, ny):
                    cost = math.sqrt(dx**2 + dy**2)
                    neighbors.append(((nx, ny), cost))
        return neighbors

    def heuristic(self, cell_a, cell_b):
        (x1, y1) = cell_a
        (x2, y2) = cell_b
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def find_closest_boundary_target(self, goal_world_x, goal_world_y):
        if not self.current_grid:
            return None

        width = self.current_grid.info.width
        height = self.current_grid.info.height
        best_target_cell = None
        min_dist_sq = float('inf')

        potential_targets = []

        # Iterate over boundary cells
        for x in range(width):
            potential_targets.append((x, 0))
            potential_targets.append((x, height - 1))
        for y in range(1, height - 1):
            potential_targets.append((0, y))
            potential_targets.append((width - 1, y))

        for cell_x, cell_y in potential_targets:
            if self.is_valid_cell(cell_x, cell_y):
                world_x, world_y = self.grid_to_world(cell_x, cell_y)
                dist_sq = (world_x - goal_world_x)**2 + (world_y - goal_world_y)**2

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_target_cell = (cell_x, cell_y)

        if best_target_cell:
            return best_target_cell
        else:
            return None
        
    def find_reachable_goal_near(self, goal_world_x, goal_world_y, top_k):
        if not self.current_grid or not self.current_grid.data:
            return None

        width = self.current_grid.info.width
        height = self.current_grid.info.height

        start_cell_x, start_cell_y = self.world_to_grid(goal_world_x, goal_world_y)

        if not (0 <= start_cell_x < width and 0 <= start_cell_y < height):
            start_cell_x, start_cell_y = self.find_closest_boundary_target(goal_world_x, goal_world_y)

        queue = collections.deque([(start_cell_x, start_cell_y)])
        visited = set([(start_cell_x, start_cell_y)])
        max_iterations = width * height
        iterations = 0
        all_potential_targets = []

        while queue and iterations < max_iterations:
            iterations += 1
            curr_x, curr_y = queue.popleft()

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    next_x, next_y = curr_x + dx, curr_y + dy
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue

                    if (next_x, next_y) not in visited:
                        visited.add((next_x, next_y))
                        state = self.current_grid.data[next_y * width + next_x]

                        # Found the top-k free cell during the outward search
                        if state == 0:
                            if len(all_potential_targets) < top_k:
                                dist_sq = (next_x - start_cell_x)**2 + (next_y - start_cell_y)**2
                                heapq.heappush(all_potential_targets, (dist_sq, (next_x, next_y)))

                            # Return all k closest free cells
                            if len(all_potential_targets) >= top_k:
                                return all_potential_targets

                        queue.append((next_x, next_y))

        return None


    def a_star_search(self, start_cell, target_cell):
        if not self.current_grid or not start_cell or not target_cell:
            return None

        open_set = []
        heapq.heappush(open_set, (self.heuristic(start_cell, target_cell), 0, start_cell))

        came_from = {}
        g_cost = {start_cell: 0}

        while open_set:
            current_f, current_g, current_cell = heapq.heappop(open_set)

            if current_g > g_cost.get(current_cell, float('inf')):
                continue

            if current_cell == target_cell:
                return self.reconstruct_path(came_from, current_cell)

            # Explore neighbors
            for neighbor_cell, move_cost in self.get_neighbors(current_cell[0], current_cell[1]):
                width = self.current_grid.info.width
                index = neighbor_cell[1] * width + neighbor_cell[0]

                state = self.current_grid.data[index]
                # This should not happen as unknown cells are not traversable during navigation
                if self.current_grid.data[index] == -1:
                    move_cost += 1000.0
                tentative_g_cost = current_g + move_cost

                if tentative_g_cost < g_cost.get(neighbor_cell, float('inf')):
                    came_from[neighbor_cell] = current_cell
                    g_cost[neighbor_cell] = tentative_g_cost
                    f_cost = tentative_g_cost + self.heuristic(neighbor_cell, target_cell)
                    heapq.heappush(open_set, (f_cost, tentative_g_cost, neighbor_cell))

        return None, None, None

    def reconstruct_path(self, came_from, current_cell):
        path = [current_cell]
        path_length_grid = 0.0
        unknown_count = 0

        while current_cell in came_from:
            prev_cell = came_from[current_cell]
            dx = current_cell[0] - prev_cell[0]
            dy = current_cell[1] - prev_cell[1]
            path_length_grid += math.sqrt(dx**2 + dy**2)

            width = self.current_grid.info.width
            index = prev_cell[1] * width + prev_cell[0]
            if self.current_grid.data[index] == -1:
                unknown_count += 1

            current_cell = prev_cell
            path.append(current_cell)
        path_length_world = path_length_grid * self.current_grid.info.resolution
        return path[::-1], path_length_world, unknown_count

    def publish_empty_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.current_grid.header.frame_id
        self.path_pub.publish(path_msg)


    def publish_path(self, grid_path):
        if not grid_path or not self.current_grid:
            return

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.current_grid.header.frame_id

        for cell in grid_path:
            world_x, world_y = self.grid_to_world(cell[0], cell[1])
            if world_x is None or world_y is None:
                 continue

            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = self.config.AGENT_HEIGHT
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


if __name__ == '__main__':
    try:
        graph_path = rospy.get_param('/graph_planner/graph_path', None)
        if graph_path is not None:
            graph_path = [ast.literal_eval(point) for point in graph_path]

        PATH_LENGTH_FACTOR = 5
        UNKNOWN_CELL_THRESHOLD = 0.5
        planner = MidPlanner(graph_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass