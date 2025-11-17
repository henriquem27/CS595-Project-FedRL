import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import umap
# --- Configuration ---
NPZ_FILE = 'federated_training_data.npz'
REWARD_SMOOTHING_WINDOW = 100  # Window size for rolling average





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

def plot_learning_curves(data):
    """
    Plots the smoothed episode rewards for each client over training steps.
    """
    print("Generating Learning Curve plot...")

    # Create a pandas DataFrame for easier manipulation
    df = pd.DataFrame({
        'steps': data['ep_steps'],
        'reward': data['ep_rewards'],
        'label': data['ep_labels']
    })

    plt.figure(figsize=(12, 7))

    # Get unique labels (e.g., 'Client_1_Standard', 'Client_2_Masked')
    labels = df['label'].unique()

    for label in labels:
        # Filter data for the current client
        client_df = df[df['label'] == label].sort_values(by='steps')

        if client_df.empty:
            print(f"Warning: No episode data found for '{label}'.")
            continue

        # Calculate the smoothed reward
        # .rolling() creates a rolling window
        # .mean() calculates the average reward in that window
        client_df['smoothed_reward'] = client_df['reward'].rolling(
            window=REWARD_SMOOTHING_WINDOW,
            min_periods=1
        ).mean()

        # Plot the smoothed reward
        plt.plot(
            client_df['steps'],
            client_df['smoothed_reward'],
            label=f'{label} (Smoothed, window={REWARD_SMOOTHING_WINDOW})'
        )

        # Optional: Plot the raw, unsmoothed data
        # plt.scatter(client_df['steps'], client_df['reward'], alpha=0.1, s=5, label=f'{label} (Raw)')

    plt.title('Federated Learning: Smoothed Episode Rewards')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('fl_learning_curves.png')
    print("Saved 'learning_curves.png'")


# ===============================================
#  Plot 2: t-SNE of Model Weights
# ===============================================

def plot_tsne_weights(data):
    """
    Performs t-SNE on the high-dimensional weight vectors and plots them
    in 2D, colored by training step and using different markers for clients.
    """
    print("\nGenerating t-SNE plot (this may take a minute)...")

    weights = data['weights']
    labels = data['weight_labels']
    steps = data['weight_steps']

    if weights.shape[0] == 0:
        print("Error: No weight data found to plot.")
        return

    # 1. Run t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(weights)

    # 2. Create the scatter plot
    plt.figure(figsize=(12, 10))

    # --- MODIFICATION: Create marker mapping ---
    unique_labels = sorted(list(set(labels)))
    # List of available markers
    markers_list = ['.', 's', '^', 'o', 'D', '<', '8', 'd', '>']
    # Map each unique label to a marker
    label_to_marker = {label: markers_list[i % len(markers_list)]
                       for i, label in enumerate(unique_labels)}

    # We no longer need the label_to_color map for the legend
    # colors = plt.cm.get_cmap('jet', len(unique_labels))
    # label_to_color = {label: colors(i)
    #                   for i, label in enumerate(unique_labels)}
    # --- END MODIFICATION ---

    # Plot points one by one
    for label in unique_labels:
        indices = np.where(labels == label)[0]

        if indices.size == 0:
            continue

        client_tsne_results = tsne_results[indices]
        client_steps = steps[indices]

        # Get the short display name
        display_name = get_display_label(label)

        # --- MODIFICATION: Add the 'marker' argument ---
        sc = plt.scatter(
            client_tsne_results[:, 0],
            client_tsne_results[:, 1],
            c=client_steps,         # Color by training step
            cmap='plasma_r',
            label=display_name,
            marker=label_to_marker[label],  # <-- Use the client's marker
            alpha=0.7,
            s=60,  # size of points (made slightly larger)
            vmin=np.min(steps),
            vmax=np.max(steps)
        )
        # --- END MODIFICATION ---

    plt.title('t-SNE Visualization of Model Weights')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

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
    # Changed title to 'Clients (Markers)'
    plt.legend(handles, legend_display_names, title='Clients (Markers)')
    # --- END MODIFICATION ---

    # Add a colorbar to show what the colors (steps) mean
    cbar = plt.colorbar(sc)
    cbar.set_label('Training Timestep')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('fl_tsne_weights_markers.png')  # Saved to a new filename
    print("Saved 'tsne_weights_markers.png'")
# ===============================================
#  Main execution
# ===============================================


def plot_umap_weights(data):
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

    plt.title('UMAP Visualization of Model Weights')
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
    plt.savefig('fl_umap_weights_markers.png')  # Saved to a new filename
    print("Saved 'umap_weights_markers.png'")


def plot_umap_weights_snapshot(data):
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
    plt.savefig('fl_umap_weights_snapshot_grid.png', bbox_inches='tight')
    print("Saved 'umap_weights_snapshot_grid.png'")


def main():
    try:
        # Load the compressed data file
        data = np.load(NPZ_FILE)
        print(f"Loaded data from '{NPZ_FILE}'. Contains:")
        print(f"  {list(data.keys())}")
        print_timestep_info(data)
        # --- Generate Plots ---
        plot_learning_curves(data)
        plot_tsne_weights(data)
        plot_umap_weights(data)
        plot_umap_weights_snapshot(data)

        print("\nAll plots generated. Check for .png files in the directory.")
        plt.show()  # Optional: display plots interactively

    except FileNotFoundError:
        print(f"Error: Could not find the file '{NPZ_FILE}'.")
        print("Please make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
