import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import umap
# --- Configuration ---
REWARD_SMOOTHING_WINDOW = 100  # Window size for rolling average

directory = "plots"
import os   



def print_timestep_info(data):
    """
    Prints a summary of the available timesteps for rewards and weights.
    """
    print("\n--- Timestep Information ---")

    try:
        if 'ep_steps' in data and len(data['ep_steps']) > 0:
            ep_steps = data['ep_steps']
            unique_ep_steps = np.unique(ep_steps)
            print("Episode Reward Timesteps:")
            print(f"  Total data points: {len(ep_steps)}")
            print(f"  Range: {np.min(ep_steps)} to {np.max(ep_steps)}")
            print(f"  {len(unique_ep_steps)} Unique timesteps recorded:")
            # Use array2string for a clean, single-line printout
            print(f"    {np.array2string(unique_ep_steps, separator=', ')}")
        else:
            print("No episode reward timestep data found.")

        print("-" * 20)  # Separator

        if 'weight_steps' in data and len(data['weight_steps']) > 0:
            weight_steps = data['weight_steps']
            unique_weight_steps = np.unique(weight_steps)
            print("Model Weight Timesteps:")
            print(f"  Total data points: {len(weight_steps)}")
            print(f"  Range: {np.min(weight_steps)} to {np.max(weight_steps)}")
            print(f"  {len(unique_weight_steps)} Unique timesteps recorded:")
            print(f"    {np.array2string(unique_weight_steps, separator=', ')}")
        else:
            print("No model weight timestep data found.")

    except Exception as e:
        print(f"Error while printing timestep info: {e}")

    print("--- End of Timestep Information ---\n")

def get_display_label(full_label):
    """
    Extracts the 'type' from a label like 'Client_X_Type'.
    e.g., 'Client_1_Standard' -> 'Standard'
    """
    try:
        return full_label.split('_', 2)[2]
    except IndexError:
        return full_label


PPO_BUFFER_SIZE = 2048

# --- Add this helper function somewhere above plot_learning_curves ---


def get_client_type(label_str):
    """
    Extracts the environment 'type' from a label like 'Client_1_Moon'.
    """
    try:
        # Assumes format 'Prefix_ID_Type'
        return label_str.split('_', 2)[2]
    except (IndexError, TypeError):
        # Fallback for unexpected formats
        return label_str

# --- Replace your old function with this one ---


def plot_learning_curves(data, output_filename,title):
    """
    Plots the smoothed episode rewards over Local Client Epochs (Updates).
    Groups clients of the same type by color.
    """
    print("Generating Learning Curve plot (X-axis: Local Epochs)...")

    df = pd.DataFrame({
        'steps': data['ep_steps'],
        'reward': data['ep_rewards'],
        'label': data['ep_labels']
    })

    # --- NEW: Create color mapping based on client type ---
    # 1. Get the type for every entry
    df['type'] = df['label'].apply(get_client_type)

    # 2. Find all unique types (e.g., ['Earth', 'Mars', 'Moon'])
    unique_types = sorted(df['type'].unique())

    # 3. Create a color map (e.g., {'Earth': 'blue', 'Mars': 'orange', ...})
    # We use a standard colormap ('tab10') to get distinct colors
    colors = plt.cm.get_cmap('tab10', len(unique_types))
    color_map = {type_name: colors(i)
                 for i, type_name in enumerate(unique_types)}

    # --- NEW: Track which types we've added to the legend ---
    legend_added = set()

    # --- Use a larger figure for better legend placement ---
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = df['label'].unique()

    for label in labels:
        client_df = df[df['label'] == label].sort_values(by='steps')

        if client_df.empty:
            print(f"Warning: No episode data found for '{label}'.")
            continue

        client_df['local_epoch'] = client_df['steps'] / PPO_BUFFER_SIZE
        client_df['smoothed_reward'] = client_df['reward'].rolling(
            window=REWARD_SMOOTHING_WINDOW,
            min_periods=1
        ).mean()

        # --- NEW: Get type, color, and label for this plot ---
        client_type = client_df['type'].iloc[0]  # Get this client's type
        plot_color = color_map.get(client_type, 'gray')  # Get its color

        plot_label = None  # Default to no legend entry
        if client_type not in legend_added:
            # If this is the first time we see this type, add it to the legend
            plot_label = client_type
            legend_added.add(client_type)

        # --- MODIFIED: Use new color, label, and alpha ---
        ax.plot(
            client_df['local_epoch'],
            client_df['smoothed_reward'],
            label=plot_label,      # Will be 'Moon' or None
            color=plot_color,      # Will be the mapped color
            alpha=0.6              # Add transparency to see overlapping lines
        )

    ax.set_title(f'{title}: Reward over Local Updates')
    ax.set_xlabel(f'Local Client Epochs (1 Epoch = {PPO_BUFFER_SIZE} steps)')
    ax.set_ylabel('Smoothed Reward')
    ax.set_ylim(-400, 400)

    # --- MODIFIED: Simpler legend, placed inside the plot ---
    ax.legend(loc='best', title="Environment Type")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(directory, output_filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(path)
    print("Saved 'fl_learning_curves_epochs.png'")

# ===============================================
#  Plot 2: t-SNE of Model Weights
# ===============================================


def plot_tsne_weights(data, output_filename,title):
    """
    Performs t-SNE on the high-dimensional weight vectors and plots them.
    Automatically adjusts perplexity for small datasets.
    """
    print("\nGenerating t-SNE plot...")

    weights = data['weights']
    labels = data['weight_labels']

    # Use 'weight_epochs' if available (for consistent coloring across methods),
    # otherwise fallback to 'weight_steps'
    if 'weight_epochs' in data and len(data['weight_epochs']) > 0:
        steps = data['weight_epochs']
        cbar_label = 'Training Epoch (Round)'
    else:
        steps = data['weight_steps']
        cbar_label = 'Training Timestep'

    n_samples = weights.shape[0]
    if n_samples < 2:
        print("Error: Not enough weight data to plot (need at least 2 samples).")
        return

    # --- FIX: Dynamic Perplexity ---
    # Perplexity must be < n_samples.
    # Standard TSNE default is 30.
    # We set it to min(30, n_samples - 1) to avoid the crash.
    perplexity_val = min(30, n_samples - 1)

    print(
        f"Running t-SNE with perplexity={perplexity_val} (n_samples={n_samples})...")

    # 1. Run t-SNE
    tsne = TSNE(n_components=2, verbose=1,
                perplexity=perplexity_val, max_iter=1000)
    tsne_results = tsne.fit_transform(weights)

    # 2. Create the scatter plot
    plt.figure(figsize=(12, 10))

    unique_labels = sorted(list(set(labels)))
    markers_list = ['.', 's', '^', 'o', 'D', '<', '8', 'd', '>']
    label_to_marker = {label: markers_list[i % len(markers_list)]
                       for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            continue

        client_tsne_results = tsne_results[indices]
        client_steps = steps[indices]
        display_name = get_display_label(label)

        sc = plt.scatter(
            client_tsne_results[:, 0],
            client_tsne_results[:, 1],
            c=client_steps,
            cmap='plasma_r',
            label=display_name,
            marker=label_to_marker[label],
            alpha=0.7,
            s=60,
            vmin=np.min(steps),
            vmax=np.max(steps)
        )

    plt.title(f'{title}:t-SNE Visualization of Model Weights (n={n_samples})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Legend
    handles = [plt.Line2D([0], [0], marker=label_to_marker[label], color='w',
                          markerfacecolor='grey', markeredgecolor='black', markersize=10)
               for label in unique_labels]
    plt.legend(handles, [get_display_label(l)
               for l in unique_labels], title='Clients')

    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(directory, output_filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(path)
    print(f"Saved '{output_filename}'")



def plot_reward_vs_epoch(data,output_filename,title):
    """
    Plots the AVERAGE episode reward for each client over EPOCHS (FL Rounds).
    """
    print("Generating Reward vs Epoch plot...")

    # 1. Check if epoch data exists

    # 2. Create DataFrame
    df = pd.DataFrame({
        'epoch': data['ep_epochs'],
        'reward': data['ep_rewards'],
        'label': data['ep_labels']
    })

    plt.figure(figsize=(12, 7))

    labels = df['label'].unique()

    for label in labels:
        # Filter for client
        client_df = df[df['label'] == label]

        # --- CRITICAL STEP: Aggregate by Epoch ---
        # We take the mean of all episodes that finished within the same epoch
        epoch_stats = client_df.groupby('epoch')['reward'].mean().reset_index()

        # Sort just in case
        epoch_stats = epoch_stats.sort_values(by='epoch')

        # Plot
        plt.plot(
            epoch_stats['epoch'],
            epoch_stats['reward'],
            marker='o',       # Add dots to mark each round clearly
            markersize=4,
            label=f'{get_display_label(label)}'
        )

    plt.title(f'{title}: Average Reward per Round (Epoch)')
    plt.xlabel('Federated Learning Epoch (Round)')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(directory, output_filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(path)
    print("Saved 'fl_reward_vs_epoch.png'")

def plot_umap_weights(data,output_filename,title):
    """
    Performs UMAP on the high-dimensional weight vectors and plots them
    in 2D, colored by training step and using different markers for clients.
    """
    print("\nGenerating UMAP plot (this may take a minute)...")

    weights = data['weights']
    labels = data['weight_labels']
    steps = data['weight_steps']

    if weights.shape[0] == 0:
        print("Error: No weight data found to plot.")
        return

    # 1. Run UMAP
    # verbose=True provides progress updates like TSNE's verbose=1
    reducer = umap.UMAP(n_components=2, verbose=True)
    umap_results = reducer.fit_transform(weights)

    # 2. Create the scatter plot
    plt.figure(figsize=(12, 10))

    # --- Create marker mapping ---
    unique_labels = sorted(list(set(labels)))
    # List of available markers
    markers_list = ['.', 's', '^', 'o', 'D', '<', '8','d','>']
    # Map each unique label to a marker
    label_to_marker = {label: markers_list[i % len(markers_list)]
                       for i, label in enumerate(unique_labels)}

    # Plot points one by one
    for label in unique_labels:
        indices = np.where(labels == label)[0]

        if indices.size == 0:
            continue

        client_umap_results = umap_results[indices]
        client_steps = steps[indices]

        # Get the short display name
        display_name = get_display_label(label)

        # --- Plot with 'marker' argument ---
        sc = plt.scatter(
            client_umap_results[:, 0],
            client_umap_results[:, 1],
            c=client_steps,         # Color by training step
            cmap='plasma_r',
            label=display_name,
            marker=label_to_marker[label],  # <-- Use the client's marker
            alpha=0.7,
            s=60,  # size of points
            vmin=np.min(steps),
            vmax=np.max(steps)
        )

    plt.title(f'{title}:UMAP Visualization of Model Weights')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # Create a custom legend for client labels
    handles = [plt.Line2D([0], [0],
                          marker=label_to_marker[label],  # Show correct marker
                          color='w',  # White line (invisible)
                          markerfacecolor='grey',  # Use a neutral color
                          markeredgecolor='black',
                          markersize=10)
               for label in unique_labels]

    legend_display_names = [get_display_label(
        label) for label in unique_labels]
    plt.legend(handles, legend_display_names, title='Clients (Markers)')

    # Add a colorbar to show what the colors (steps) mean
    cbar = plt.colorbar(sc)
    cbar.set_label('Training Timestep')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(directory, output_filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(path)  # Saved to a new filename
    print("Saved 'umap_weights_markers.png'")


def plot_umap_weights_snapshot(data,output_filename):
    """
    Performs a *separate* UMAP run for *each* unique timestep and plots
    each result in a subplot.

    WARNING: This approach is statistically limited. UMAP requires a
    reasonable number of data points (e_g. > 15) to produce a
    meaningful topological structure. Running it on just a few
    clients for a single timestep will likely produce unstable or
    uninformative plots.
    """
    print("\nGenerating UMAP snapshot grid (one UMAP per timestep)...")
    print("  WARNING: This plot may be uninformative if there are few clients per timestep.")

    weights = data['weights']
    labels = data['weight_labels']
    steps = data['weight_steps']

    if weights.shape[0] == 0:
        print("Error: No weight data found to plot.")
        return

    # 1. Create a DataFrame for easier filtering
    df = pd.DataFrame({
        'step': steps,
        'label': labels
    })

    # 2. Get all unique timesteps
    unique_timesteps = np.sort(df['step'].unique())
    n_plots = len(unique_timesteps)

    if n_plots == 0:
        print("Error: No unique timesteps found in weight data.")
        return

    # 3. Create color and marker maps for clients
    unique_labels = sorted(np.unique(labels))
    markers_list = ['o', 's', '^', 'v', 'P', 'X', '*']
    colors_list = plt.cm.get_cmap('jet', len(unique_labels))

    label_to_marker = {label: markers_list[i % len(markers_list)]
                       for i, label in enumerate(unique_labels)}
    label_to_color = {label: colors_list(i)
                      for i, label in enumerate(unique_labels)}

    # 4. Create the subplot grid
    # Aim for a 4-column layout
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))

    # Increase figure size to accommodate all plots
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 4)
        # NOTE: We DO NOT sharex/sharey. Each UMAP is a new coordinate system.
    )
    axes = axes.flatten()

    # 5. Loop, run UMAP, and plot for each timestep
    for i, ax in enumerate(axes):

        if i >= n_plots:
            ax.axis('off')  # Turn off extra subplots
            continue

        current_timestep = unique_timesteps[i]
        ax.set_title(f"Timestep: {current_timestep}")

        # Get indices for *only* this timestep
        indices_to_use = (df['step'] == current_timestep).values

        current_weights = weights[indices_to_use]
        current_labels = labels[indices_to_use]

        n_samples = current_weights.shape[0]

        # 6. Check if we have enough data to run UMAP
        # UMAP needs n_neighbors < n_samples. Default n_neighbors=15.
        # We must set it dynamically.
        if n_samples < 3:
            ax.text(0.5, 0.5, 'Not enough data\n( < 3 samples)',
                    ha='center', va='center', transform=ax.transAxes)
            print(
                f"Skipping timestep {current_timestep}: only {n_samples} samples.")
            continue

        # Dynamically set n_neighbors to be less than n_samples
        # n_neighbors must be at least 2
        n_neighbors = max(2, min(15, n_samples - 1))

        print(
            f"Running UMAP for T={current_timestep} ({n_samples} samples, n_neighbors={n_neighbors})...")

        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,  # Use 0.1 for more clustering
                verbose=False
            )
            embedding = reducer.fit_transform(current_weights)

            # 7. Plot this embedding on its subplot (ax)
            plot_df = pd.DataFrame({
                'umap_1': embedding[:, 0],
                'umap_2': embedding[:, 1],
                'label': current_labels
            })

            for label in unique_labels:
                client_df = plot_df[plot_df['label'] == label]
                if client_df.empty:
                    continue

                ax.scatter(
                    client_df['umap_1'],
                    client_df['umap_2'],
                    marker=label_to_marker[label],
                    color=[label_to_color[label]],  # 'c' expects an array
                    label=get_display_label(label),
                    alpha=0.8,
                    s=60
                )
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        except Exception as e:
            ax.text(0.5, 0.5, 'UMAP Error', ha='center',
                    va='center', transform=ax.transAxes)
            print(f"Error running UMAP for timestep {current_timestep}: {e}")

    # 8. Create a single, shared legend
    handles = [plt.Line2D([0], [0],
                          marker=label_to_marker[label],
                          color=label_to_color[label],
                          linestyle='None',
                          markersize=10)
               for label in unique_labels]
    legend_display_names = [get_display_label(
        label) for label in unique_labels]

    fig.legend(
        handles,
        legend_display_names,
        title='Clients',
        loc='upper right',
        bbox_to_anchor=(0.99, 0.98)
    )

    fig.suptitle('UMAP Snapshot per Timestep (Independent Runs)',
                 fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])  # Adjust for legend
    path = os.path.join(directory, output_filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(path)
    print("Saved 'umap_weights_snapshot_grid.png'")


def main():
    NPZ_FILE = 'training_data.npz'
    try:
        # Load the compressed data file
        data = np.load(NPZ_FILE)
        print(f"Loaded data from '{NPZ_FILE}'. Contains:")
        print(f"  {list(data.keys())}")
        # print_timestep_info(data)
        # --- Generate Plots ---
        plot_learning_curves(data)
        plot_tsne_weights(data)
        plot_umap_weights(data)
        # plot_reward_vs_epoch(data)

        print("\nAll plots generated. Check for .png files in the directory.")
        plt.show()  # Optional: display plots interactively

    except FileNotFoundError:
        print(f"Error: Could not find the file '{NPZ_FILE}'.")
        print("Please make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
