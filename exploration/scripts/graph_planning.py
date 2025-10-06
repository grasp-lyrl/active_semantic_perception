#!/usr/bin/env python3

import rospy
import random
import heapq
from collections import defaultdict
import networkx as nx
import argparse
import numpy as np
import os
import spark_dsg as dsg
import yaml
import ast
import datetime
import json

class Node:
    def __init__(self, name, node_type, pos, id):
        self.name = name
        self.type = node_type
        self.parent = None
        self.children = set()
        self.position = pos
        self.id = id

class GraphPlanner:
    def __init__(self, config):
        """
        Initializes the graph planner.
        """
        self.config = config
        self.graph = {}
        self.room_connections = defaultdict(set)

    def load_graph(self, filename):
        with open(filename, 'r') as file:
            graph_data = yaml.safe_load(file)
        nodes_by_id = {node['id']: node for node in graph_data['nodes']}
        for edge in graph_data['edges']:
            source_id, target_id = edge[0], edge[1]

            # Get the full data for both nodes in the edge
            source_node = nodes_by_id[source_id]
            target_node = nodes_by_id[target_id]

            # Get the node types
            source_type = source_node.get('node_type')
            target_type = target_node.get('node_type')

            # Case 1: Edge connects an object to a room
            if source_type == 'object' and target_type == 'room':
                if source_id not in self.graph:
                    # Register the node to graph planner's graph
                    self.graph[source_id] = Node(source_node['name'], 'object', str(source_node['position']), source_id)
                if target_id not in self.graph:
                    self.graph[target_id] = Node(target_node['name'], 'room', str(target_node['position']), target_id)

                self.graph[source_id].parent = self.graph[target_id]
                self.graph[target_id].children.add(self.graph[source_id])

            # Case 2: Edge connects a room to an object
            elif source_type == 'room' and target_type == 'object':
                if source_id not in self.graph:
                    self.graph[source_id] = Node(source_node['name'], 'room', str(source_node['position']), source_id)
                if target_id not in self.graph:
                    self.graph[target_id] = Node(target_node['name'], 'object', str(target_node['position']), target_id)

                self.graph[target_id].parent = self.graph[source_id]
                self.graph[source_id].children.add(self.graph[target_id])

            # Case 3: Edge connects two rooms
            elif source_type == 'room' and target_type == 'room':
                if source_id not in self.graph:
                    self.graph[source_id] = Node(source_node['name'], 'room', str(source_node['position']), source_id)
                if target_id not in self.graph:
                    self.graph[target_id] = Node(target_node['name'], 'room', str(target_node['position']), target_id)

                self.connect_rooms(source_id, target_id)


    def connect_rooms(self, r1, r2):
        self.room_connections[r1].add((r2, 1))
        self.room_connections[r2].add((r1, 1))

    def nearest_object_node(self, current_pos):
        all_object_pos = [(n.id, ast.literal_eval(n.position)) for n in self.graph.values() if n.type == 'object']
        distances = [((current_pos[0] - obj_pos[0])**2 + (current_pos[1] - obj_pos[1])**2 + (current_pos[2] - obj_pos[2])**2)**0.5 for _,obj_pos in all_object_pos]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        nearest_object = all_object_pos[min_index]
        return self.graph[nearest_object[0]]
    
    def nearest_room_node(self, current_pos):
        all_room_pos = [(n.id, ast.literal_eval(n.position)) for n in self.graph.values() if n.type == 'room']
        distances = [((current_pos[0] - room_pos[0])**2 + (current_pos[1] - room_pos[1])**2 + (current_pos[2] - room_pos[2])**2)**0.5 for _,room_pos in all_room_pos]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        nearest_room = all_room_pos[min_index]
        return self.graph[nearest_room[0]]

    def a_star(self, start_room, goal_room):
        open_set = []
        heapq.heappush(open_set, (0, start_room))
        came_from = {}
        g_score = {start_room: 0}
        f_score = {start_room: self.heuristic(start_room, goal_room)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_room:
                return self.reconstruct_path(came_from, current)

            for neighbor, cost in self.room_connections[current]:
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_room)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def heuristic(self, room1, room2):
        return 0

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path
    
    def plan(self, scene_graph_file, start_pos, target_pose):
        """
        Plan a path on the scene graph
        """
        self.graph.clear()
        self.room_connections.clear()

        # 1. Load the scene graph of rooms and objects
        self.load_graph(scene_graph_file)

        # 2. Find the nearest rooms to the start and target positions
        target_pos = target_pose[:3]
        start_room = self.nearest_room_node(start_pos)
        target_room = self.nearest_room_node(target_pos)

        # 3. Find the path of rooms using A* search
        room_id_path = self.a_star(start_room.id, target_room.id)
        
        room_path = [self.graph[r_id].name for r_id in room_id_path]
        room_path_pos = [self.graph[r_id].position for r_id in room_id_path]

        # 4. Construct the final list of waypoints
        # The path starts at the agent's current position, goes through the centers
        # of the intermediate rooms, and ends at the final target position.
        if start_room.id != target_room.id:
            full_path = ["start_position"] + room_path + ["semantic uncertain region"]
            full_path_pos = [start_pos] + room_path_pos + [target_pos]
        else: # If start and end are in the same room
            full_path = ["start_position", "semantic uncertain region"]
            full_path_pos = [start_pos, target_pos]

        # rospy.loginfo("Planned path:")
        # for step in full_path:
        #     rospy.loginfo(f" -> {step}")
        # rospy.loginfo("Planned path positions:")
        # for pos in full_path_pos:
        #     rospy.loginfo(f" -> {pos}")
        rospy.loginfo(f"Selected Viewpoint: {full_path_pos[-1]}")

        final_path = [np.array(ast.literal_eval(pos) if isinstance(pos, str) else pos) for pos in full_path_pos]

        data_to_save = {
            "timestamp": datetime.datetime.now().isoformat(),
            "final_path": [path.tolist() for path in final_path],
            "final_path_name": full_path
        }

        json_filepath = os.path.join(self.config.WORKING_DIRECTORY, "path_log.json")
        with open(json_filepath, "a") as f:
            f.write(json.dumps(data_to_save) + "\n")
        return final_path

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Graph Planner')
    # TODO(huayi): think of combining multiple graphs for planning?
    default_path = "/home/apple/Work/scene_graph_processing/Report_Experiment_Office_2/Step_6/habitat_scene_graph_new.graphml"
    rospy.set_param('/scene_graph_path', default_path)
    arg_parser.add_argument('--path', type=str, default=default_path, help='Path to the graphml file')
    args = arg_parser.parse_args()                                   
    GraphPlanner(args.path)
    rospy.spin()
