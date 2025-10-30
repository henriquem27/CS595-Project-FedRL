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
    in 2D, colored by client and training step.
    """
    print("\nGenerating t-SNE plot (this may take a minute)...")

    weights = data['weights']
    labels = data['weight_labels']
    steps = data['weight_steps']

    if weights.shape[0] == 0:
        print("Error: No weight data found to plot.")
        return


    # 1. Run t-SNE
    # n_components=2 means we reduce to 2 dimensions
    # NEW CORRECTED LINE
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(weights)

    # 2. Create the scatter plot
    plt.figure(figsize=(12, 10))

    # Map labels to colors
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('jet', len(unique_labels))
    label_to_color = {label: colors(i)
                      for i, label in enumerate(unique_labels)}

    # Plot points one by one to color them by step
    # We use 'steps' to create a color gradient from light to dark
    for label in unique_labels:
        # Get indices for this specific client
        indices = np.where(labels == label)[0]

        if indices.size == 0:
            continue

        client_tsne_results = tsne_results[indices]
        client_steps = steps[indices]

        # Scatter plot for this client
        # 'c=client_steps' maps the step number to a color
        # 'cmap='plasma_r'' is a colormap (reversed plasma)
        # 'vmin' and 'vmax' ensure the colormap spans all steps
        sc = plt.scatter(
            client_tsne_results[:, 0],
            client_tsne_results[:, 1],
            c=client_steps,
            # 'plasma_r' goes from light (start) to dark (end)
            cmap='plasma_r',
            label=label,
            alpha=0.7,
            s=50,  # size of points
            vmin=np.min(steps),
            vmax=np.max(steps)
        )

    plt.title('t-SNE Visualization of Model Weights')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Create a custom legend for client labels
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=label_to_color[label], markersize=10)
               for label in unique_labels]
    plt.legend(handles, unique_labels, title='Clients')

    # Add a colorbar to show what the colors (steps) mean
    cbar = plt.colorbar(sc)
    cbar.set_label('Training Timestep')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('tsne_weights.png')
    print("Saved 'tsne_weights.png'")


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
