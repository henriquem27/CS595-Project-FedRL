import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

# --- Configuration ---
NPZ_FILE = 'federated_training_data.npz'
REWARD_SMOOTHING_WINDOW = 100  # Window size for rolling average

# ===============================================
#  Plot 1: Learning Curves (Smoothed Rewards)
# ===============================================


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
    plt.savefig('learning_curves.png')
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
    markers_list = ['o', 's', '^', 'v', 'P', 'X', '*']
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

    # --- MODIFICATION: Update legend to show markers ---
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
    plt.savefig('tsne_weights_markers.png')  # Saved to a new filename
    print("Saved 'tsne_weights_markers.png'")
# ===============================================
#  Main execution
# ===============================================

def main():
    try:
        # Load the compressed data file
        data = np.load(NPZ_FILE)
        print(f"Loaded data from '{NPZ_FILE}'. Contains:")
        print(f"  {list(data.keys())}")

        # --- Generate Plots ---
        plot_learning_curves(data)
        plot_tsne_weights(data)

        print("\nAll plots generated. Check for .png files in the directory.")
        plt.show()  # Optional: display plots interactively

    except FileNotFoundError:
        print(f"Error: Could not find the file '{NPZ_FILE}'.")
        print("Please make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
