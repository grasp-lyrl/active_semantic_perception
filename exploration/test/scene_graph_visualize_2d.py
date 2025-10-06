import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import itertools
import yaml
import os
import numpy as np
import matplotlib.transforms as transforms


def visualize_scene_for_z_range(data, z_range, output_filename):
    """
    Generates a 2D top-down visualization of a scene graph for a specific
    range of Z coordinates.
    """
    min_z, max_z = z_range

    # --- 1. Data Processing ---
    def parse_vector(s):
        if isinstance(s, str):
            return json.loads(s)
        return s

    nodes_by_id = {node['id']: node for node in data['nodes']}

    all_objects = []
    for node in data['nodes']:
        if node['node_type'] in ['object', 'nothing', 'structure']:
            all_objects.append({
                'id': node['id'],
                'name': node['name'],
                'center': parse_vector(node['position']),
                'dims': parse_vector(node['dimension']),
                'orientation': parse_vector(node['orientation']) if 'orientation' in node else np.eye(3),
                'room_id': None if node['node_type'] == 'object' else 'nothing',
                'type': node['node_type']
            })
            
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for obj in all_objects:
        center_x, center_y, _ = obj['center']
        width, height, _ = obj['dims']
        min_x = min(min_x, center_x - width / 2)
        max_x = max(max_x, center_x + width / 2)
        min_y = min(min_y, center_y - height / 2)
        max_y = max(max_y, center_y + height / 2)
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    global_x_lim = (min_x - padding_x, max_x + padding_x)
    global_y_lim = (min_y - padding_y, max_y + padding_y)

    for obj in all_objects:
        if obj['room_id'] == 'nothing': continue
        for edge in data['edges']:
            if edge[0] == obj['id']:
                obj['room_id'] = edge[1]
                break
            elif edge[1] == obj['id']:
                obj['room_id'] = edge[0]
                break

    # --- 2. Filter Objects by Z-Position --- 
    objects_in_slice = []
    for obj in all_objects:
        center_z = obj['center'][2]
        z_dimension = obj['dims'][2]
        obj_min_z = center_z - z_dimension / 2
        obj_max_z = center_z + z_dimension / 2

        if (obj_min_z < max_z and obj_max_z > min_z) or obj['type'] == 'structure' or obj['name'] == 'door':
            objects_in_slice.append(obj)

    # --- 3. Visualization Setup ---
    room_ids = sorted(list(set(node['id'] for node in data['nodes'] if node['node_type'] == 'room')))
    color_palette = ['#3b82f6', '#10b981', '#ef4444', '#f97316', '#8b5cf6', '#ec4899', '#f59e0b', '#6366f1']
    color_cycle = itertools.cycle(color_palette)
    room_colors = {room_id: next(color_cycle) for room_id in room_ids}
    fig = plt.figure(figsize=(18, 12))
    ax = fig.gca()

    # --- 4. Draw Filtered Objects with Orientation ---

    for obj in objects_in_slice:
        width, height, z_dim = obj['dims']
        center_x, center_y, center_z = obj['center']
        room_color = room_colors.get(obj['room_id'], "#888888")

        rot_matrix = obj['orientation']

        yaw_radians = np.arctan2(rot_matrix[1][0], rot_matrix[0][0])
        yaw_degrees = np.degrees(yaw_radians)
        
        if obj['name'] == 'door':
            rect = patches.Rectangle(
            (-width / 2, -height / 2), width, height,
            linewidth=1.5, edgecolor='black', facecolor=room_color, alpha=0.8, zorder=2
        )
        else:
            rect = patches.Rectangle(
                (-width / 2, -height / 2), width, height,
                linewidth=1.5, edgecolor='black', facecolor=room_color, alpha=0.8
            )

        transform = transforms.Affine2D() \
            .rotate_deg(yaw_degrees) \
            .translate(center_x, center_y)
            
        rect.set_transform(transform + ax.transData)
        ax.add_patch(rect)

        ax.text(
            center_x, center_y, f"{obj['name'].capitalize()}",
            ha='center', va='center', fontsize=9, fontweight='bold', color='black'
        )
        
    # --- 5. Configure and Finalize Plot ---
    ax.set_xlim(global_x_lim)
    ax.set_ylim(global_y_lim)
    
    visible_room_ids = set(obj['room_id'] for obj in objects_in_slice if obj['room_id'])
    legend_patches = []
    for room_id, color in room_colors.items():
        if room_id in visible_room_ids:
            room_name_raw = nodes_by_id[room_id]['name']
            room_name = room_name_raw.replace("an image of a ", "").capitalize()
            legend_patches.append(patches.Patch(color=color, label=f"{room_name} (ID: {room_id})"))
    
    if legend_patches:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.title(f'2D Scene Visualization (Z-Range: {min_z:.1f} to {max_z:.1f})', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)


def create_scene_slices(scene_graph_file, scene_slice_folder_name):
    # Define the Z-ranges for slicing the scene
    z_ranges = [
        (0.0, 0.3),
        (0.3, 0.5),
        (0.5, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
    ]

    # Load the scene graph data from the YAML file
    with open(scene_graph_file, 'r') as file:
        sample_data = yaml.safe_load(file)

    # Create an output directory if it doesn't exist
    parent_dir = os.path.dirname(scene_graph_file)
    output_dir = os.path.join(parent_dir, scene_slice_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generate a separate plot for each defined Z-range
    for min_z, max_z in z_ranges:
        filename = f"scene_z_{str(min_z).replace('.', '_')}_to_{str(max_z).replace('.', '_')}.jpg"
        full_path = os.path.join(output_dir, filename)

        visualize_scene_for_z_range(sample_data, (min_z, max_z), full_path)
    

if __name__ == "__main__":
    scene_graph_file = "/home/apple/Work/scene_graph_processing/Summer_research/Claude_Test_2/habitat_scene_graph_original.yaml"
    create_scene_slices(scene_graph_file)