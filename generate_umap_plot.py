import os
import glob
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_and_process_weights(base_dir):
    data = []
    
    # Pattern to extract round number from directory name
    round_pattern = re.compile(r'round_(\d+)')
    
    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        # Check if current directory is a round directory
        dir_name = os.path.basename(root)
        match = round_pattern.match(dir_name)
        
        if match:
            round_num = int(match.group(1))
            print(f"Processing Round {round_num}...")
            
            for file in files:
                if file.endswith(".pt"):
                    file_path = os.path.join(root, file)
                    client_id = file.replace('.pt', '')
                    
                    try:
                        # Load weights
                        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
                        
                        # Flatten weights
                        weights_flat = []
                        for key in sorted(state_dict.keys()):
                            # Skip if not a parameter (e.g., batch norm stats if any, though PPO MLP usually just weights/biases)
                            # For safety, just take everything in state_dict that is a tensor
                            param = state_dict[key]
                            if isinstance(param, torch.Tensor):
                                weights_flat.append(param.view(-1).numpy())
                        
                        if weights_flat:
                            weights_vector = np.concatenate(weights_flat)
                            data.append({
                                'round': round_num,
                                'client_id': client_id,
                                'weights': weights_vector
                            })
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    return pd.DataFrame(data)

def get_client_type(client_id):
    """Extracts client type (e.g., Moon, Mars, Earth) from client_id."""
    if "Moon" in client_id: return "Moon"
    if "Mars" in client_id: return "Mars"
    if "Earth" in client_id: return "Earth"
    return "Unknown"

def main():
    base_dir = "/Users/henriquerio/Documents/IIT/CS595-Project-FedRL/sv_results/v4/logs/fl_run/weights"
    
    print("Loading weights...")
    df = load_and_process_weights(base_dir)
    
    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} samples.")
    
    # Extract weight matrix
    weight_matrix = np.stack(df['weights'].values)
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, verbose=False, random_state=42)
    embedding = reducer.fit_transform(weight_matrix)
    
    df['umap_x'] = embedding[:, 0]
    df['umap_y'] = embedding[:, 1]
    
    print("Plotting...")
    
    # --- User's Custom Plotting Logic ---
    output_filename = "umap_weights_custom.png"
    
    weights = weight_matrix # Not used directly in plotting but part of user's logic context
    labels = df['client_id'].values
    steps = df['round'].values
    umap_results = embedding

    # 1. Determine Unique Types for Marker Mapping
    all_types = [get_client_type(l) for l in labels]
    unique_types = sorted(list(set(all_types)))
    
    markers_list = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '<', '>']
    type_to_marker = {t: markers_list[i % len(markers_list)] 
                      for i, t in enumerate(unique_types)}

    # 3. Plot
    plt.figure(figsize=(12, 10))
    unique_client_labels = sorted(list(set(labels)))

    sc = None
    for label in unique_client_labels:
        indices = np.where(labels == label)[0]
        if indices.size == 0: continue

        client_umap_results = umap_results[indices]
        client_steps = steps[indices]

        # Determine type for marker
        current_type = get_client_type(label)
        marker = type_to_marker.get(current_type, 'o')

        sc = plt.scatter(
            client_umap_results[:, 0],
            client_umap_results[:, 1],
            c=client_steps,
            cmap='plasma_r',
            marker=marker, # <--- Marker based on Type
            alpha=0.7,
            s=60,
            vmin=np.min(steps),
            vmax=np.max(steps)
        )

    plt.title('UMAP of Model Weights (Shape=Type, Color=Time)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # 4. Create Custom Legend for Types
    handles = []
    legend_labels = []
    for t in unique_types:
        h = plt.Line2D([0], [0], marker=type_to_marker[t], color='w',
                       markerfacecolor='grey', markeredgecolor='black', 
                       markersize=10)
        handles.append(h)
        legend_labels.append(t)

    plt.legend(handles, legend_labels, title='Agent Types')

    if sc:
        cbar = plt.colorbar(sc)
        cbar.set_label('Training Timestep (Round)')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"Saved '{output_filename}'")

if __name__ == "__main__":
    main()
