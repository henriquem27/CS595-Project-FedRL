import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# --- Configuration ---
FED_FILE = 'federated_training_data.npz'
SINGLE_FILE = 'training_data.npz'
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

    # 1. Load and combine data
    fed_df = load_and_tag_data(FED_FILE, 'Federated')
    single_df = load_and_tag_data(SINGLE_FILE, 'Single')

    if fed_df.empty and single_df.empty:
        print("No data to plot. Exiting.")
        return

    all_df = pd.concat([fed_df, single_df])

    # 2. Align data by binning timesteps
    all_df['aligned_step'] = (all_df['steps'] // BIN_SIZE) * BIN_SIZE

    # 3. Set up plot colors and styles
    plt.figure(figsize=(14, 8))

    unique_labels = sorted(all_df['label'].unique())
    unique_experiments = sorted(all_df['experiment'].unique())

    # Map each client (Moon, Earth...) to a consistent color
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Map each experiment (Fed, Single) to a consistent line style
    style_map = {
        'Federated': {'linestyle': '-', 'marker': 'o', 'markersize': 3},
        'Single':    {'linestyle': '--', 'marker': 'x', 'markersize': 4}
    }

    # 4. Plot the data
    # We loop by CLIENT first, then experiment, to assign colors correctly
    for label in unique_labels:
        client_df = all_df[all_df['label'] == label]
        color = color_map[label]

        for exp_type in unique_experiments:
            exp_client_df = client_df[client_df['experiment'] == exp_type]
            if exp_client_df.empty:
                continue

            # Aggregate data for this specific line
            aligned_df = exp_client_df.groupby('aligned_step')[
                'reward'].mean().reset_index()

            # Get plot style
            style = style_map.get(exp_type, {'linestyle': ':', 'marker': 's'})
            plot_label_name = f"{get_display_label(label)} ({exp_type})"

            plt.plot(
                aligned_df['aligned_step'],
                aligned_df['reward'],
                label=plot_label_name,
                color=color,
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=style['markersize'],
                linewidth=2
            )

    plt.title(
        f'Federated vs. Single Training Comparison (Aligned every {BIN_SIZE} steps)')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Average Reward')
    # Legend outside plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

    output_filename = 'comparison_learning_curves.png'
    path = os.path.join(directory, output_filename)
    
    plt.savefig(path)
    print(f"Saved comparison plot to '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    plot_combined_learning_curves()
