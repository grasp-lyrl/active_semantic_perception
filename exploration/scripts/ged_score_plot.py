import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import yaml
import json
import sys
import ast
import time
from gmatch4py.ged.graph_edit_dist import GraphEditDistance
import itertools

def load_graph_from_yaml(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Graph file not found at {file_path}", file=sys.stderr)
        return None

    with open(file_path, 'r') as f:
        scene_graph_data = yaml.safe_load(f)

    G = nx.Graph()
    nodes = scene_graph_data.get('nodes', [])
    edges = scene_graph_data.get('edges', [])

    # Add nodes with their attributes
    for node_data in nodes:
        if node_data['name'] != 'door': # Skip 'door' nodes
            if node_data['node_type'] == 'object' or node_data['node_type'] == 'room':
                node_id = node_data['id']
                # Keep all attributes, especially 'name' for matching
                attributes = {k: v for k, v in node_data.items() if k != 'id'}
                G.add_node(node_id, **attributes)

    # Add edges
    for edge_data in edges:
        if len(edge_data) == 2:
            u, v = edge_data
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v)

    for node_id, node_data in G.nodes(data=True):
        if node_data.get('node_type') == 'object':
            parent_name = None
            # Find the parent room among its neighbors
            for neighbor_id in G.neighbors(node_id):
                neighbor_data = G.nodes[neighbor_id]
                if neighbor_data.get('node_type') == 'room':
                    parent_name = neighbor_data.get('name')
                    break
            G.nodes[node_id]['parent_name'] = parent_name

    return G

def smart_eval(expr):
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return expr

def custom_node_cost(node1_attrs, node2_attrs, object_threshold=0.5, room_threshold=4.0):
    """
    Calculates the substitution cost using different distance thresholds
    for objects and rooms.
    """
    node_type = node1_attrs.get('node_type')

    # 1. Forbid matching if types or names are different.
    if node_type != node2_attrs.get('node_type'):
        return float('inf')
    if node1_attrs.get('name') != node2_attrs.get('name'):
        return float('inf')
    
    if node_type == 'object':
        if node1_attrs.get('parent_name') != node2_attrs.get('parent_name'):
            return float('inf') # Forbid match if parents differ

    # 2. Parse position strings.
    pos1 = np.array(smart_eval(node1_attrs.get('position', '[0,0,0]')))
    pos2 = np.array(smart_eval(node2_attrs.get('position', '[0,0,0]')))
    
    # 3. Calculate Euclidean distance.
    difference_vector = pos1 - pos2
    if node_type == 'room':
        difference_vector[2] = 0
    distance = np.linalg.norm(difference_vector)
    
    # 4. Apply the correct threshold based on the node type.
    if node_type == 'object':
        if distance > object_threshold:
            return float('inf') # Object is too far.
    elif node_type == 'room':
        if distance > room_threshold:
            return float('inf') # Room is too far.

    # 5. If there is a match, the cost is just zero
    return 0

class CustomGED(GraphEditDistance):
    # Redefine the substitute_cost (This is the core part)
    def substitute_cost(self, node1, node2, G, H):
        node1_attributes = G.nodes[node1]
        node2_attributes = H.nodes[node2]
        
        return custom_node_cost(node1_attributes, node2_attributes)

def main():
    """
    Main function to calculate and plot Graph Edit Distance (GED).
    """
    # --- Configuration ---
    gt_folder = '/home/apple/Work/scene_graph_processing/Summer_research/GT_00871_2'
    pipelines = [
        '/home/apple/Work/scene_graph_processing/Summer_research/Frontier_00871',
        '/home/apple/Work/scene_graph_processing/Summer_research/Semantic_Exploration_00871',
        '/home/apple/Work/scene_graph_processing/Summer_research/SSMI_Pipeline_Test_00871']
    results = {pipeline: [] for pipeline in pipelines}

    # --- Ground Truth Data Loading ---
    gt_yaml_file = os.path.join(gt_folder, 'habitat_scene_graph_original_graph0.yaml')
    print(f"Loading ground truth graph from: {gt_yaml_file}")
    gt_graph = load_graph_from_yaml(gt_yaml_file)
    if not gt_graph:
        print("Could not load the ground truth graph. Exiting.", file=sys.stderr)
        return

    # --- Pipeline Data Processing ---
    print("\n--- Starting GED Calculation for Pipelines ---")
    for pipeline in pipelines:
        pipeline_folder = pipeline
        if not os.path.isdir(pipeline_folder):
            print(f"Warning: Pipeline directory not found: {pipeline_folder}", file=sys.stderr)
            continue
        
        print(f"Processing pipeline: {os.path.basename(pipeline)}")
        
        stages = sorted([ast.literal_eval(d) for d in os.listdir(pipeline_folder)])
        
        for stage in stages:
            print(f"  - Stage {stage}")
            stage_folder = os.path.join(pipeline, str(stage))
            path_length_file = os.path.join(stage_folder, 'navigation_stats.json')
            graph_yaml_file = os.path.join(stage_folder, 'habitat_scene_graph_original_graph0.yaml')

            if not os.path.exists(graph_yaml_file) or not os.path.exists(path_length_file):
                print(f"  - Warning: Missing YAML or path_length file in {stage_folder}. Skipping.", file=sys.stderr)
                continue
            
            # Load path length for the current stage
            with open(path_length_file, 'r') as f:
                path_data = json.load(f)
                path_length = path_data.get("total_path_length_meters")
                nav_time = path_data.get("total_navigation_time_seconds")

            if path_length is None:
                print(f"  - Warning: 'path_length' key not found in {path_length_file}. Skipping.", file=sys.stderr)
                continue

            if path_length == 0:
                predicted_graph = nx.Graph()
            else:
                # Load the graph for the current stage
                predicted_graph = load_graph_from_yaml(graph_yaml_file)
                if not predicted_graph:
                    print(f"  - Warning: Could not load graph from {graph_yaml_file}. Skipping.", file=sys.stderr)
                    continue

            # Calculate Graph Edit Distance (GED)
            # This computes the minimum cost of operations (node/edge changes)
            # to transform the predicted_graph into the gt_graph.
            # A lower GED means the graphs are more similar.
            start_time = time.time()
            GED=CustomGED(1,1,1,1)
            ged = GED.distance_ged(predicted_graph, gt_graph)

            end_time = time.time()
            print(f"  - Stage {stage}: Path Length={path_length:.2f}m, GED={ged}, Time={end_time - start_time:.2f}s")

            results[pipeline].append({'path_length': path_length, 'ged': ged, 'stage': stage, 'time': nav_time})

    # --- Plotting Results ---
    plt.figure(figsize=(12, 7))
    for pipeline, data_points in results.items():
        if not data_points: continue
        
        # Sort data by path_length to ensure the line plots correctly
        sorted_data = sorted(data_points, key=lambda x: x['path_length'])

        x_values = [d['path_length'] for d in sorted_data]
        x_path_length = [d['path_length'] for d in sorted_data]
        y_values = [d['ged'] for d in sorted_data]

        # Determine the label for the plot legend
        if 'Semantic' in pipeline:
            label = 'Semantic Exploration'
        elif 'Frontier' in pipeline:
            label = 'Frontier Exploration'
        elif 'SSMI' in pipeline:
            label = 'SSMI Exploration'
        else:
            label = os.path.basename(pipeline)
            
        plt.plot(x_values, y_values, linestyle='-', label=label, linewidth=4)

    plt.xlabel('Path Length (m)', fontsize=20)
    plt.xlim(0, 165)
    plt.ylabel('Graph Edit Distance (GED)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('Graph Edit Distance Comparison of Pipelines (00573)', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig('/home/apple/Work/exp_pipeline_ws/ged_analysis_00871_v2.png')

    print("\n--- Graph Edit Distance Results ---")
    for pipeline, data in results.items():
        pipeline_name = os.path.basename(pipeline)
        print(f"\nPipeline: {pipeline_name}")
        if data:
            for point in sorted(data, key=lambda x: x['path_length']):
                print(f"  Path Length: {point['path_length']:.2f}m -> GED={point['ged']:.2f}")
        else:
            print("  No data found.")
            
    print(" Plot saved as ged_analysis.png")

if __name__ == '__main__':
    main()
