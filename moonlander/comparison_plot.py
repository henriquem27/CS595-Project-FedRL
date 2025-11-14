import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# --- Configuration ---
FED_FILE = 'federated_training_data.npz'
SINGLE_FILE = 'training_data.npz'
DP_FILE = 'dp_training_data.npz'
BIN_SIZE = 1000  # Aligns data to every 1000 steps
# ---------------------
import os
directory = "plots"
def get_display_label(full_label):
    """
    Extracts the 'type' from a label like 'Client_X_Type'.
    e.g., 'Client_1_Moon' -> 'Moon'
    """
    try:
        # Split only twice, take the third part
        return full_label.split('_', 2)[2]
    except IndexError:
        return full_label


def load_and_tag_data(filepath, experiment_tag):
    """Loads an NPZ file and tags its data with the experiment type."""
    try:
        data = np.load(filepath)
        df = pd.DataFrame({
            'steps': data['ep_steps'],
            'reward': data['ep_rewards'],
            'label': data['ep_labels']
        })
        df['experiment'] = experiment_tag
        print(f"Loaded {len(df)} episode points from '{filepath}'")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'. Skipping.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading '{filepath}': {e}. Skipping.")
        return pd.DataFrame()


def plot_combined_learning_curves():
    """
    Plots learning curves from Federated and Single runs on the same graph
    for direct comparison.
    """
    print("Generating combined learning curve comparison plot...")

    # Create plots directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # 1. Load and combine data
    fed_df = load_and_tag_data(FED_FILE, 'Federated')
    single_df = load_and_tag_data(SINGLE_FILE, 'Single')
    dp_df = load_and_tag_data(DP_FILE, 'DP')

    all_df = pd.concat([fed_df, single_df, dp_df])

    if all_df.empty:
        print("No data to plot. Exiting.")
        return

    # 2. Align data by binning timesteps
    all_df['aligned_step'] = (all_df['steps'] // BIN_SIZE) * BIN_SIZE

    # 3. Set up plot colors and styles
    plt.figure(figsize=(14, 8))

    unique_labels = sorted(all_df['label'].unique())
    unique_experiments = sorted(all_df['experiment'].unique())

    # --- SWAP 1: Define maps based on the new logic ---

    # Map each experiment (Fed, Single, DP) to a consistent color
    # Using hardcoded colors for clarity is often better
    color_map = {
        'Federated': '#0072B2',  # Blue
        'Single':    '#009E73',  # Green
        'DP':        '#D55E00',  # Red/Orange
    }
    # Add a default color for any unexpected experiment types
    default_color = '#AAAAAA'  # Grey

    # Map each client (Moon, Earth...) to a consistent line style/marker
    markers = ['o', 'X', 's', '^', 'v', 'D', 'P']  # List of markers
    linestyles = ['-', '--', ':', '-.']  # List of linestyles

    style_map = {}
    for i, label in enumerate(unique_labels):
        style_map[label] = {
            'linestyle': linestyles[i % len(linestyles)],
            'marker': markers[i % len(markers)],
            'markersize': 5
        }
    # Add a default style for any unexpected labels
    default_style = {'linestyle': ':', 'marker': '.', 'markersize': 4}

    # --- End of SWAP 1 ---

    # 4. Plot the data
    # We loop by CLIENT first, then experiment (original loop order is fine)
    for label in unique_labels:
        client_df = all_df[all_df['label'] == label]

        # --- SWAP 2: Get style from label ---
        style = style_map.get(label, default_style)

        for exp_type in unique_experiments:
            exp_client_df = client_df[client_df['experiment'] == exp_type]
            if exp_client_df.empty:
                continue

            # Aggregate data for this specific line
            aligned_df = exp_client_df.groupby('aligned_step')[
                'reward'].mean().reset_index()

            # --- SWAP 2: Get color from experiment type ---
            color = color_map.get(exp_type, default_color)

            plot_label_name = f"{get_display_label(label)} ({exp_type})"

            plt.plot(
                aligned_df['aligned_step'],
                aligned_df['reward'],
                label=plot_label_name,
                color=color,  # <-- Apply experiment color
                linestyle=style['linestyle'],  # <-- Apply client style
                marker=style['marker'],  # <-- Apply client marker
                markersize=style['markersize'],  # <-- Apply client marker size
                linewidth=2
            )
    # --- End of SWAP 2 ---

    plt.title(
        f'Training Comparison (Aligned every {BIN_SIZE} steps)')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Average Reward')
    plt.ylim(-1500, 500)
    # Legend outside plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

    output_filename = 'comparison_learning_curves_by_experiment.png'
    path = os.path.join(directory, output_filename)

    plt.savefig(path)
    print(f"Saved comparison plot to '{path}'")
    plt.show()


def plot_sidebyside_curves():
    """
    Plots learning curves from Federated and Single runs on the same graph
    for direct comparison.
    """
    print("Generating combined learning curve comparison plot...")

    # Create plots directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # 1. Load and combine data
    fed_df = load_and_tag_data(FED_FILE, 'Federated')
    single_df = load_and_tag_data(SINGLE_FILE, 'Single')
    dp_df = load_and_tag_data(DP_FILE, 'DP')

    all_df = pd.concat([fed_df, single_df, dp_df])

    if all_df.empty:
        print("No data to plot. Exiting.")
        return

    # 2. Align data by binning timesteps
    all_df['aligned_step'] = (all_df['steps'] // BIN_SIZE) * BIN_SIZE

    # 3. Set up plot colors and styles
    plt.figure(figsize=(14, 8))

    unique_labels = sorted(all_df['label'].unique())
    unique_experiments = sorted(all_df['experiment'].unique())

    n_clients = len(unique_labels)
    if n_clients == 0:
        print("No client data found to plot.")
        return

    # --- Create Subplots: 1 row, n_clients columns ---
    # Plots will share X and Y axes for direct comparison
    fig, axes = plt.subplots(1, n_clients, figsize=(
        6 * n_clients, 7), sharey=True, sharex=True)

    # Ensure 'axes' is always an array, even if n_clients is 1
    if n_clients == 1:
        axes = [axes]
    # --------------------------------------------------

    # --- SWAP 1: Define maps based on the new logic ---

    # Map each experiment (Fed, Single, DP) to a consistent color
    # Using hardcoded colors for clarity is often better
    color_map = {
        'Federated': '#0072B2',  # Blue
        'Single':    '#009E73',  # Green
        'DP':        '#D55E00',  # Red/Orange
    }
    # Add a default color for any unexpected experiment types
    default_color = '#AAAAAA'  # Grey

    # Map each client (Moon, Earth...) to a consistent line style/marker
    # We will map EXPERIMENT type to style now, since client is the subplot
    style_map = {
        'Federated': {'linestyle': '-', 'marker': 'o', 'markersize': 4},
        'Single':    {'linestyle': '--', 'marker': 'X', 'markersize': 5},
        'DP':        {'linestyle': ':', 'marker': 's', 'markersize': 4},
    }
    # Add a default style for any unexpected labels
    default_style = {'linestyle': ':', 'marker': '.', 'markersize': 4}

    # --- End of SWAP 1 ---

    # 4. Plot the data
    # We loop by CLIENT first, and assign each client to a subplot
    for i, label in enumerate(unique_labels):
        ax = axes[i]  # Get the correct subplot axis
        client_df = all_df[all_df['label'] == label]

        for exp_type in unique_experiments:
            exp_client_df = client_df[client_df['experiment'] == exp_type]
            if exp_client_df.empty:
                continue

            # Aggregate data for this specific line
            aligned_df = exp_client_df.groupby('aligned_step')[
                'reward'].mean().reset_index()

            # --- SWAP 2: Get color and style from experiment type ---
            color = color_map.get(exp_type, default_color)
            style = style_map.get(exp_type, default_style)

            plot_label_name = exp_type  # Label is just 'Federated', 'Single'

            ax.plot(
                aligned_df['aligned_step'],
                aligned_df['reward'],
                label=plot_label_name,
                color=color,  # <-- Apply experiment color
                linestyle=style['linestyle'],  # <-- Apply client style
                marker=style['marker'],  # <-- Apply client marker
                markersize=style['markersize'],  # <-- Apply client marker size
                linewidth=2
            )
        # --- End of SWAP 2 ---

        # --- Set subplot-specific properties ---
        ax.set_title(f"Client: {get_display_label(label)}")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Training Timesteps')
        # -------------------------------------

    # 5. Set shared figure properties

    # --- Set Y-axis limits (applies to all due to sharey=True) ---
    axes[0].set_ylim(-1500, 1500)

    # --- Set shared Y-axis label ---
    axes[0].set_ylabel('Average Reward')

    # --- Set shared main title ---
    fig.suptitle(
        f'Training Comparison (Aligned every {BIN_SIZE} steps)', fontsize=16)

    # --- Create a single shared legend at the bottom ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(
        0.5, 0.01), ncol=len(unique_experiments), fontsize=12)

    # Adjust layout to make room for suptitle and legend
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0.02, 0.1, 0.98, 0.92])

    output_filename = 'comparison_learning_curves_subplots.png'
    path = os.path.join(directory, output_filename)

    plt.savefig(path)
    print(f"Saved comparison plot to '{path}'")
    plt.show()

if __name__ == '__main__':
    plot_combined_learning_curves()
    plot_sidebyside_curves()