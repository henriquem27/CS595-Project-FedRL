import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# --- Configuration ---
OUTPUT_DIR = "comparison_plots"

# File names for each experiment
DATA_FILES = {
    'Single Agent': 'training_data.npz',
    'Standard FL': 'federated_training_data.npz',
    'DP FL epsilon=5, sensitivity=15': 'dp_training_data_ep5_sens15.npz',
    'DP FL epsilon=10, sensitivity=15': 'dp_training_data_ep10_sens15.npz',
    'DP FL epsilon=30, sensitivity=15': 'dp_training_data_ep15_sens15.npz',

}

# How to group timesteps for smoothing
# (e.g., 2000 means 0-1999, 2000-3999, etc.)
STEP_BIN_SIZE = 2000
# Set a consistent X-axis limit
MAX_STEPS = 200000
# Set a consistent Y-axis limit for rewards
REWARD_YLIM = (-400, 300)

# --- Plotting Mappings ---
# Color will represent the *Training Method*
COLOR_MAP = {
    'Single Agent': '#2ca02c',  # Green
    'Standard FL': '#1f77b4',  # Blue
    'DP FL epsilon=5, sensitivity=15': '#d62728',
    'DP FL epsilon=10, sensitivity=15': '#ff7f0e',
    'DP FL epsilon=30, sensitivity=15': '#2ca02c',

}

# Line Style will represent the *Environment*
STYLE_MAP = {
    'Moon': 'solid',
    'Mars': 'dashed',
    'Earth': 'dotted'
}

# --- Helper Functions ---


def load_data(filename):
    """Loads an .npz file if it exists."""
    if not os.path.exists(filename):
        print(f"Warning: Data file not found: {filename}")
        return None
    try:
        return np.load(filename, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def get_client_type(label_str):
    """Extracts 'Moon', 'Earth', or 'Mars' from a label."""
    try:
        return label_str.split('_')[-1]
    except Exception:
        return 'unknown'


def process_data(data):
    """
    Converts raw .npz data into an aggregated Pandas DataFrame.
    Calculates mean/min/max per step_bin and type.
    """
    if data is None:
        return pd.DataFrame()

    df = pd.DataFrame({
        'steps': data['ep_steps'],
        'reward': data['ep_rewards'],
        'label': data['ep_labels']
    })

    # Add environment type column (Moon, Earth, Mars)
    df['type'] = df['label'].apply(get_client_type)

    # Bin the steps to create smoothed groups
    df['step_bin'] = (df['steps'] // STEP_BIN_SIZE) * STEP_BIN_SIZE

    # Group by the binned steps and env type, then calculate stats
    grouped = df.groupby(['step_bin', 'type'])['reward'].agg(
        mean='mean',
        min='min',
        max='max'
    ).reset_index()

    return grouped

# --- Main Plotting Function ---


def plot_master_comparison(datasets):
    """
    Generates the 4-way comparison plot.
    """
    print("Generating master comparison plot...")

    fig, ax = plt.subplots(figsize=(16, 10))  # Large figure for readability

    # Loop over each method (Single, FL, Clustered, DP)
    for method_name, grouped_data in datasets.items():
        if grouped_data.empty:
            print(f"  Skipping {method_name} (no data).")
            continue

        plot_color = COLOR_MAP.get(method_name, 'gray')

        # Loop over each env type (Moon, Mars, Earth)
        for env_type, plot_style in STYLE_MAP.items():

            # Get the specific data for this method AND env
            type_data = grouped_data[grouped_data['type'] == env_type]
            if type_data.empty:
                continue

            # 1. Plot the MEAN line
            ax.plot(
                type_data['step_bin'],
                type_data['mean'],
                color=plot_color,
                linestyle=plot_style,
                linewidth=2,
                label=f"{method_name} - {env_type}"  # Internal label
            )

            # 2. Plot the MIN/MAX shaded band
            # ax.fill_between(
            #     type_data['step_bin'],
            #     type_data['min'],
            #     type_data['max'],
            #     color=plot_color,
            #     alpha=0.1,  # Very transparent
            #     edgecolor='none'
            # )

    # --- Create Custom Legends (outside the plot) ---
    # We create "dummy" lines to represent the colors and styles

    # Legend 1: Colors for Training Method
    color_patches = [
        mlines.Line2D([], [], color=c, linestyle='solid', linewidth=3,
                      label=name)
        for name, c in COLOR_MAP.items() if name in datasets
    ]
    legend1 = ax.legend(
        handles=color_patches,
        title='Training Method',
        loc='upper center',
        bbox_to_anchor=(0.25, -0.1),  # Position below plot
        ncol=2,
        fontsize='large',
        title_fontsize='large'
    )

    # Legend 2: Line Styles for Environment
    style_lines = [
        mlines.Line2D([], [], color='black', linestyle=s, linewidth=2,
                      label=name)
        for name, s in STYLE_MAP.items()
    ]
    ax.legend(
        handles=style_lines,
        title='Environment Type',
        loc='upper center',
        bbox_to_anchor=(0.75, -0.1),  # Position below plot
        ncol=1,
        fontsize='large',
        title_fontsize='large'
    )

    # Add the first legend back manually (ax.legend overwrites)
    ax.add_artist(legend1)

    # --- Final Plot Setup ---
    ax.set_title('Master Comparison: Method vs. Environment', fontsize=18)
    ax.set_xlabel(
        f'Timesteps (Grouped in bins of {STEP_BIN_SIZE})', fontsize=14)
    ax.set_ylabel('Reward (Mean with Min/Max Band)', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_ylim(REWARD_YLIM)
    ax.set_xlim(0, MAX_STEPS)

    # Adjust layout to make room for the legends at the bottom
    plt.subplots_adjust(bottom=0.25)

    save_path = os.path.join(OUTPUT_DIR, "comparison_master_plot.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.show()
    print(f"\nMaster plot saved to: {save_path}")


# --- Main execution ---
if __name__ == "__main__":

    # 1. Load all datasets
    raw_datasets = {
        name: load_data(file)
        for name, file in DATA_FILES.items()
    }

    # 2. Process them into aggregated dataframes
    processed_datasets = {
        name: process_data(data)
        for name, data in raw_datasets.items()
    }

    # 3. Plot
    plot_master_comparison(processed_datasets)
