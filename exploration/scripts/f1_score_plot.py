import os
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import yaml
import ast
import spark_dsg as dsg
import spark_dsg.networkx as dsg_nx
import pathlib
from bidict import bidict
import networkx as nx
import json

def smart_eval(expr):
    return ast.literal_eval(expr) if isinstance(expr, str) else expr

def search_nearest_room_node(G, object_pos):
    room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    room_nodes = list(room_layer.nodes)
    best_parent = None
    best_distance = float('inf')
    for room_node in room_nodes:
        room_pos = room_node.attributes.position
        distance = np.linalg.norm(np.array(room_pos) - np.array(object_pos))
        if distance < best_distance:
            best_distance = distance
            best_parent = room_node.id.value
    return best_parent

def convert_dsg_to_yaml(dsg_filepath):
    base_path = dsg_filepath.parent
    output_yaml_path = os.path.join(base_path, "habitat_scene_graph_original_graph0.yaml")

    # 1. Load the scene graph
    G = dsg.DynamicSceneGraph.load(str(dsg_filepath))
    object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
    
    # Ensure objects have a room parent
    for node in object_layer.nodes:
        if node.attributes.name == "Nothing":
            node.attributes.position = node.attributes.bounding_box.world_P_center
            continue
        
        place_parent = node.get_parent()
        room_parent = None
        if place_parent is not None:
            place_parent_node = G.get_node(place_parent)
            room_parent = place_parent_node.get_parent()
            
        if room_parent is None:
            object_pos = node.attributes.position
            room_parent = search_nearest_room_node(G, object_pos)
            
        if room_parent is not None:
            attr = dsg._dsg_bindings.EdgeAttributes()
            G.insert_edge(node.id.value, room_parent, attr)

    # 2. Convert to NetworkX and process
    easier_graph = dsg_nx.graph_to_networkx(G, include_dynamic=False)
    remove_nodes = []
    mapping = bidict()
    idx = 0

    for node, attribute in easier_graph.nodes.items():
        node_prefix = str(node)[:3]
        if node_prefix in ['828', '807']:  # Segment or Place Node
            remove_nodes.append(node)
        elif node_prefix == '799':  # Object Node
            new_id = idx
            mapping[new_id] = node
            idx += 1
            attribute['position'] = attribute['bounding_box'].world_P_center
            attribute['dimension'] = attribute['bounding_box'].dimensions
            attribute['orientation'] = attribute['bounding_box'].world_R_center
            if attribute['name'] == 'Nothing':
                attribute['node_type'] = 'nothing'
            elif attribute['name'] in ["Wall", "Window", "Curtain", "Blind", "Door"]:
                attribute['node_type'] = 'structure'
            else:
                attribute['node_type'] = 'object'
        elif node_prefix == '778':  # Room Node
            new_id = idx
            mapping[new_id] = node
            idx += 1
            attribute['node_type'] = 'room'
        else:
            raise ValueError(f"Unexpected node prefix for node {node}: {node_prefix}")
        
        # Clean attributes
        for key in list(attribute.keys()):
            if key not in ['position', 'node_type', 'name', 'is_predicted', 'dimension', 'orientation']:
                del attribute[key]
            elif isinstance(attribute.get(key), np.ndarray):
                py_list = attribute[key].tolist()
                rounded_list = [[round(num, 3) for num in row] for row in py_list] if py_list and isinstance(py_list[0], list) else [round(x, 3) for x in py_list]
                attribute[key] = str(rounded_list)

    for node in remove_nodes:
        easier_graph.remove_node(node)
    easier_graph = nx.relabel_nodes(easier_graph, dict(mapping.inverse))

    # 3. Prepare data for YAML export
    nodes_for_yaml = [{'id': node_id, **attributes} for node_id, attributes in easier_graph.nodes(data=True)]
    edges_for_yaml = [list(edge) for edge in easier_graph.edges()]
    data_to_dump = {'nodes': nodes_for_yaml, 'edges': edges_for_yaml}

    # 4. Write the YAML file
    with open(output_yaml_path, 'w') as yaml_file:
        yaml.safe_dump(data_to_dump, yaml_file, default_flow_style=False, sort_keys=False)

def preprocess_folders(folders_to_scan):
    """Scans folders for dsg.json files and converts them to .yaml."""
    print("--- Starting Preprocessing Stage ---")
    for folder in folders_to_scan:
        print(f"Scanning '{folder}' for DSG json files...")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('graph0_dsg.json'):
                    dsg_path = pathlib.Path(root) / file
                    convert_dsg_to_yaml(dsg_path)
    print("--- Preprocessing Complete ---")

def parse_objects(file_path):
    objects = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return objects

    with open(file_path, 'r') as f:
        scene_graph = yaml.safe_load(f)

    # TODO(huayi): Door should be structure nodes.
    for node in scene_graph.get('nodes', []):
        if node.get('node_type') == 'object' and node.get('name') != 'door':
            pos = np.array(smart_eval(node['position']), dtype=float)
            object_dict = {
                'name': node.get('name'),
                'position': list(pos)
            }
            objects.append(object_dict)

    return objects


def calculate_f1_score(gt_objects, explored_objects, threshold=0.5):
    """Calculates the F1 score for object detection."""
    if not gt_objects:
        print("Warning: Ground truth contains no objects.", file=sys.stderr)
        return 0, 0, 0
    if not explored_objects:
        return 0, 0, 0 # F1=0, Precision=0, Recall=0

    true_positives = 0
    available_gt_indices = list(range(len(gt_objects)))

    # Iterate through each predicted object
    for explored_obj in explored_objects:
        best_match_idx = -1
        min_dist = float('inf')

        # Find the best match among available ground truth objects of the same name
        indices_to_check = list(available_gt_indices) # Iterate over a copy
        for gt_idx in indices_to_check:
            gt_obj = gt_objects[gt_idx]
            if explored_obj['name'] == gt_obj['name']:
                dist = np.linalg.norm(np.array(explored_obj['position']) - np.array(gt_obj['position']))
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = gt_idx
        
        # If a match is found within the threshold, count it and remove it from the pool
        if min_dist < threshold:
            true_positives += 1
            available_gt_indices.remove(best_match_idx)

    false_positives = len(explored_objects) - true_positives
    false_negatives = len(gt_objects) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall


def main():
    gt_folder = '/home/apple/Work/scene_graph_processing/Summer_research/GT_00871_2'
    pipelines = ['/home/apple/Work/scene_graph_processing/Summer_research/Frontier_00871', 
                 '/home/apple/Work/scene_graph_processing/Summer_research/Semantic_Exploration_00871',
                 '/home/apple/Work/scene_graph_processing/Summer_research/SSMI_Pipeline_Test_00871']
    results = {pipeline: [] for pipeline in pipelines}
    preprocess_folders([gt_folder])

    # --- Ground Truth Data Loading ---
    gt_file = os.path.join(gt_folder, 'habitat_scene_graph_original_graph0.yaml')
    print(f"Loading ground truth from: {gt_file}")
    gt_objects = parse_objects(gt_file)
    if not gt_objects:
        print("Could not find ground truth objects. Exiting.", file=sys.stderr)
        return

    # --- Pipeline Data Processing ---
    for pipeline in pipelines:
        pipeline_folder = pipeline
        if not os.path.isdir(pipeline_folder):
            print(f"Warning: Pipeline directory not found: {pipeline_folder}", file=sys.stderr)
            continue
            
        # Find stage subdirectories
        stages = sorted([d for d in os.listdir(pipeline_folder) if d.isdigit() and os.path.isdir(os.path.join(pipeline_folder, d))])
        
        for stage in stages:
            stage_folder = os.path.join(pipeline, stage)
            path_length_file = os.path.join(stage_folder, 'navigation_stats.json')
            yaml_file = os.path.join(stage_folder, 'habitat_scene_graph_original_graph0.yaml')

            if not os.path.exists(yaml_file) or not os.path.exists(path_length_file):
                print(f"Warning: Missing YAML or path_length file in {stage_folder}. Skipping.", file=sys.stderr)
                continue
            
            with open(path_length_file, 'r') as f:
                path_data = json.load(f)
                path_length = path_data.get("total_path_length_meters")
                time = path_data.get("total_navigation_time_seconds")

            if path_length is None:
                print(f"Warning: 'path_length' key not found in {path_length_file}. Skipping.", file=sys.stderr)
                continue
            
            predicted_objects = parse_objects(yaml_file)
            f1, pr, re = calculate_f1_score(gt_objects, predicted_objects)
            results[pipeline].append({'path_length': path_length, 'f1': f1, 'precision': pr, 'recall': re, 'stage': int(stage), 'time': time})

    # --- Plotting Results ---
    plt.figure(figsize=(12, 7))
    for pipeline, data_points in results.items():
        if not data_points: continue

        # Sort data by stage to ensure the line plots correctly
        sorted_data = sorted(data_points, key=lambda x: x['stage'])
        
        x_values = [d['path_length'] for d in sorted_data]
        x_path_length = [d['path_length'] for d in sorted_data]
        temp_y_values = [d['f1'] for d in sorted_data]
        y_values = []
        max_so_far = -float('inf')
        for y in temp_y_values:
            max_so_far = max(max_so_far, y)
            y_values.append(max_so_far)
        if x_path_length[0] != 0:
            x_values = [0] + [x_value+1 for x_value in x_values]
            y_values = [0] + y_values
        elif x_path_length[1] == 0:
            y_values = [0, 0] + y_values[2:]
        else:
            y_values = [0] + y_values[1:]

        if 'Semantic' in pipeline:
            label = 'semantic exploration'
        elif 'Frontier' in pipeline:
            label = 'frontier exploration'
        elif 'SSMI' in pipeline:
            label = 'SSMI exploration'
        plt.plot(x_values, y_values, linestyle='-', label=label, linewidth=4)

    plt.xlabel('Path Length (m)', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('F1 Score Comparison of Pipelines (00871)')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.ylim(0, 0.7)
    # plt.xlim(0, 165)
    # plt.xlim(0, 91.4)
    # plt.savefig('/home/apple/Work/exp_pipeline_ws/f1_score_time_00871.png')

    print("\n--- F1 Score Results ---")
    for pipeline, data in results.items():
        print(f"\nPipeline: {pipeline}")
        if data:
            for point in sorted(data, key=lambda x: x['path_length']):
                print(f"  Path Length: {point['path_length']:.2f}m -> F1={point['f1']:.4f}, Precision={point['precision']:.4f}, Recall={point['recall']:.4f}")
        else:
            print("  No data found.")
            
    print("Plot saved as f1_score_analysis.png")

if __name__ == '__main__':
    main()