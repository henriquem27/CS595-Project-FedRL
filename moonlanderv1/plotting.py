import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import umap
import os

# --- Configuration ---
REWARD_SMOOTHING_WINDOW = 100
PPO_BUFFER_SIZE = 2048
directory = "plots"

# --- Helper Functions ---

def get_client_type(label_str):
    """
    Extracts the environment 'type' from a label like 'Client_1_Moon'.
    """
    try:
        # Assumes format 'Prefix_ID_Type' -> returns 'Moon'
        return label_str.split('_', 2)[2]
    except (IndexError, TypeError):
        return label_str

def get_display_label(full_label):
    try:
        return full_label.split('_', 2)[2]
    except IndexError:
        return full_label

def print_timestep_info(data):
    print("\n--- Timestep Information ---")
    try:
        if 'ep_steps' in data:
            print(f"Episode Data: {len(data['ep_steps'])} points.")
        if 'weight_steps' in data:
            print(f"Weight Data: {len(data['weight_steps'])} points.")
    except Exception as e:
        print(f"Error info: {e}")
    print("--- End ---\n")

# --- Plotting Functions ---

def plot_learning_curves(data, output_filename="learning_curve.png", title="FRL"):
    print("Generating Learning Curve plot...")

    df = pd.DataFrame({
        'steps': data['ep_steps'],
        'reward': data['ep_rewards'],
        'label': data['ep_labels']
    })

    # 1. Get types and create color map
    df['type'] = df['label'].apply(get_client_type)
    unique_types = sorted(df['type'].unique())
    colors = plt.cm.get_cmap('tab10', len(unique_types))
    color_map = {t: colors(i) for i, t in enumerate(unique_types)}

    legend_added = set()
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = df['label'].unique()

    for label in labels:
        client_df = df[df['label'] == label].sort_values(by='steps')
        if client_df.empty: continue

        client_df['local_epoch'] = client_df['steps'] / PPO_BUFFER_SIZE
        client_df['smoothed_reward'] = client_df['reward'].rolling(
            window=REWARD_SMOOTHING_WINDOW, min_periods=1
        ).mean()

        client_type = client_df['type'].iloc[0]
        plot_color = color_map.get(client_type, 'gray')
        
        plot_label = client_type if client_type not in legend_added else None
        legend_added.add(client_type)

        ax.plot(
            client_df['local_epoch'],
            client_df['smoothed_reward'],
            label=plot_label,
            color=plot_color,
            alpha=0.6
        )

    ax.set_title(f'{title}: Reward over Local Updates')
    ax.set_xlabel('Local Client Epochs')
    ax.set_ylabel('Smoothed Reward')
    ax.set_ylim(-400, 400)
    ax.legend(loc='best', title="Environment Type")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, output_filename))
    print(f"Saved '{output_filename}'")

# ===============================================
#  Updated Plot 2: t-SNE (Markers by Type)
# ===============================================

def plot_tsne_weights(data, output_filename="tsne_weights.png", title="FRL"):
    print("\nGenerating t-SNE plot...")

    weights = data['weights']
    labels = data['weight_labels'] # e.g., "Client_0_Earth"

    # Determine steps for coloring
    if 'weight_epochs' in data and len(data['weight_epochs']) > 0:
        steps = data['weight_epochs']
        cbar_label = 'Training Epoch'
    else:
        steps = data['weight_steps']
        cbar_label = 'Training Timestep'

    n_samples = weights.shape[0]
    if n_samples < 2:
        print("Not enough data for t-SNE.")
        return

    # 1. Determine Unique Types for Marker Mapping
    # Extract types (e.g., 'Earth', 'Moon')
    all_types = [get_client_type(l) for l in labels]
    unique_types = sorted(list(set(all_types)))
    
    # Assign a specific marker to each unique Type
    markers_list = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '<', '>']
    type_to_marker = {t: markers_list[i % len(markers_list)] 
                      for i, t in enumerate(unique_types)}

    # 2. Run t-SNE
    perplexity_val = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_val, max_iter=1000)
    tsne_results = tsne.fit_transform(weights)

    # 3. Plot
    plt.figure(figsize=(12, 10))
    
    # Get unique client identifiers to loop through data chunks
    unique_client_labels = sorted(list(set(labels)))

    sc = None
    for label in unique_client_labels:
        indices = np.where(labels == label)[0]
        if indices.size == 0: continue

        client_tsne_results = tsne_results[indices]
        client_steps = steps[indices]
        
        # Determine the type for this specific client
        current_type = get_client_type(label)
        marker = type_to_marker.get(current_type, 'o')

        sc = plt.scatter(
            client_tsne_results[:, 0],
            client_tsne_results[:, 1],
            c=client_steps,
            cmap='plasma_r',
            marker=marker, # <--- Marker based on Type
            alpha=0.7,
            s=60,
            vmin=np.min(steps),
            vmax=np.max(steps)
        )

    plt.title(f'{title}: t-SNE of Model Weights (Shape=Type, Color=Time)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # 4. Create Custom Legend for Types (Markers)
    # We only show one legend entry per Type (Earth, Moon, etc.)
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
        cbar.set_label(cbar_label)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, output_filename))
    print(f"Saved '{output_filename}'")

# ===============================================
#  Updated Plot 3: UMAP (Markers by Type)
# ===============================================

def plot_umap_weights(data, output_filename="umap_weights.png", title="FRL"):
    print("\nGenerating UMAP plot...")

    weights = data['weights']
    labels = data['weight_labels']
    steps = data['weight_steps']

    if weights.shape[0] == 0: return

    # 1. Determine Unique Types for Marker Mapping
    all_types = [get_client_type(l) for l in labels]
    unique_types = sorted(list(set(all_types)))
    
    markers_list = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '<', '>']
    type_to_marker = {t: markers_list[i % len(markers_list)] 
                      for i, t in enumerate(unique_types)}

    # 2. Run UMAP
    reducer = umap.UMAP(n_components=2, verbose=False)
    umap_results = reducer.fit_transform(weights)

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

    plt.title(f'{title}: UMAP of Model Weights (Shape=Type, Color=Time)')
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
        cbar.set_label('Training Timestep')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, output_filename))
    print(f"Saved '{output_filename}'")

# --- Main ---

def main():
    NPZ_FILE = 'training_data.npz' # Ensure this file exists
    try:
        data = np.load(NPZ_FILE)
        print(f"Loaded data keys: {list(data.keys())}")
        
        # Note: Added filename and title arguments to match definitions
        plot_learning_curves(data, "learning_curve.png", "Federated PPO")
        plot_tsne_weights(data, "tsne_weights.png", "Federated PPO")
        plot_umap_weights(data, "umap_weights.png", "Federated PPO")

        print("\nAll plots generated.")
        # plt.show() 

    except FileNotFoundError:
        print(f"Error: '{NPZ_FILE}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()