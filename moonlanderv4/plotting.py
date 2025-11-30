"""
plotting utilities for moonlanderv4 csv-based logs.

this script reads csv files from logs/ directory and generates:
1. learning curves (reward over time)
2. comparison plots across experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from pathlib import Path

# --- configuration ---
REWARD_SMOOTHING_WINDOW = 50
PLOTS_DIR = "plots"

def load_experiment_data(experiment_path):
    """
    loads all csv metrics from an experiment directory.
    
    args:
        experiment_path: path to experiment (e.g., logs/fl_run/)
    
    returns:
        dataframe with columns: [round, step, metric, value, client_label]
    """
    metrics_dir = os.path.join(experiment_path, "metrics")
    
    if not os.path.exists(metrics_dir):
        print(f"warning: {metrics_dir} not found")
        return pd.DataFrame()
    
    all_data = []
    csv_files = glob.glob(os.path.join(metrics_dir, "*_metrics.csv"))
    
    for csv_file in csv_files:
        # extract client label from filename
        filename = os.path.basename(csv_file)
        client_label = filename.replace("_metrics.csv", "")
        
        # read csv
        df = pd.read_csv(csv_file)
        df['client_label'] = client_label
        all_data.append(df)
    
    if not all_data:
        print(f"warning: no csv files found in {metrics_dir}")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"loaded {len(combined)} data points from {len(csv_files)} clients")
    return combined


def get_client_type(label_str):
    """extracts environment type from label like 'Client_1_Moon' -> 'Moon'"""
    try:
        return label_str.split('_', 2)[2]
    except (IndexError, TypeError):
        return label_str


def plot_learning_curves(data, output_filename="learning_curves.png", title="Training Progress"):
    """
    plots smoothed learning curves for all clients.
    
    args:
        data: dataframe with columns [round, step, metric, value, client_label]
        output_filename: output file name
        title: plot title
    """
    print(f"generating learning curves: {output_filename}")
    
    # filter for episode_reward metric
    reward_data = data[data['metric'] == 'episode_reward'].copy()
    
    if reward_data.empty:
        print("warning: no episode_reward data found")
        return
    
    # add client type for coloring
    reward_data['client_type'] = reward_data['client_label'].apply(get_client_type)
    
    # get unique types and assign colors
    unique_types = sorted(reward_data['client_type'].unique())
    colors = plt.cm.get_cmap('tab10', len(unique_types))
    color_map = {t: colors(i) for i, t in enumerate(unique_types)}
    
    # create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    legend_added = set()
    
    for client_label in reward_data['client_label'].unique():
        client_data = reward_data[reward_data['client_label'] == client_label].copy()
        client_data = client_data.sort_values('step')
        
        # smooth rewards
        client_data['smoothed_reward'] = client_data['value'].rolling(
            window=REWARD_SMOOTHING_WINDOW, min_periods=1
        ).mean()
        
        client_type = client_data['client_type'].iloc[0]
        plot_color = color_map.get(client_type, 'gray')
        
        # only add to legend once per type
        plot_label = client_type if client_type not in legend_added else None
        legend_added.add(client_type)
        
        ax.plot(
            client_data['step'],
            client_data['smoothed_reward'],
            label=plot_label,
            color=plot_color,
            alpha=0.6,
            linewidth=1.5
        )
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('training steps', fontsize=12)
    ax.set_ylabel(f'smoothed episode reward (window={REWARD_SMOOTHING_WINDOW})', fontsize=12)
    ax.legend(loc='best', title="environment type", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # save
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")
    plt.close()


def plot_comparison(experiments_dict, output_filename="comparison.png", title="Experiment Comparison"):
    """
    plots comparison of multiple experiments.
    
    args:
        experiments_dict: dict of {experiment_name: data_path}
        output_filename: output file name
        title: plot title
    """
    print(f"generating comparison plot: {output_filename}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.get_cmap('tab10', len(experiments_dict))
    
    for idx, (exp_name, exp_path) in enumerate(experiments_dict.items()):
        data = load_experiment_data(exp_path)
        
        if data.empty:
            continue
        
        # filter for episode rewards
        reward_data = data[data['metric'] == 'episode_reward'].copy()
        
        if reward_data.empty:
            continue
        
        # aggregate across all clients
        aggregated = reward_data.groupby('step')['value'].mean().reset_index()
        
        # smooth
        aggregated['smoothed'] = aggregated['value'].rolling(
            window=REWARD_SMOOTHING_WINDOW, min_periods=1
        ).mean()
        
        ax.plot(
            aggregated['step'],
            aggregated['smoothed'],
            label=exp_name,
            color=colors(idx),
            linewidth=2
        )
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('training steps', fontsize=12)
    ax.set_ylabel(f'average smoothed reward (window={REWARD_SMOOTHING_WINDOW})', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # save
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")
    plt.close()


def plot_per_client_comparison(experiments_dict, output_filename="per_client_comparison.png"):
    """
    creates subplots comparing experiments for each client type.
    
    args:
        experiments_dict: dict of {experiment_name: data_path}
        output_filename: output file name
    """
    print(f"generating per-client comparison: {output_filename}")
    
    # load all data
    all_exp_data = {}
    for exp_name, exp_path in experiments_dict.items():
        data = load_experiment_data(exp_path)
        if not data.empty:
            data['client_type'] = data['client_label'].apply(get_client_type)
            all_exp_data[exp_name] = data
    
    if not all_exp_data:
        print("warning: no data loaded")
        return
    
    # get unique client types
    all_types = set()
    for data in all_exp_data.values():
        all_types.update(data['client_type'].unique())
    
    client_types = sorted(all_types)
    n_types = len(client_types)
    
    if n_types == 0:
        print("warning: no client types found")
        return
    
    # create subplots
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6), sharey=True)
    if n_types == 1:
        axes = [axes]
    
    colors = plt.cm.get_cmap('tab10', len(experiments_dict))
    
    for ax_idx, client_type in enumerate(client_types):
        ax = axes[ax_idx]
        
        for exp_idx, (exp_name, data) in enumerate(all_exp_data.items()):
            # filter for this client type and episode rewards
            filtered = data[
                (data['client_type'] == client_type) & 
                (data['metric'] == 'episode_reward')
            ].copy()
            
            if filtered.empty:
                continue
            
            # aggregate across clients of same type
            aggregated = filtered.groupby('step')['value'].mean().reset_index()
            
            # smooth
            aggregated['smoothed'] = aggregated['value'].rolling(
                window=REWARD_SMOOTHING_WINDOW, min_periods=1
            ).mean()
            
            ax.plot(
                aggregated['step'],
                aggregated['smoothed'],
                label=exp_name,
                color=colors(exp_idx),
                linewidth=2
            )
        
        ax.set_title(f"client type: {client_type}", fontsize=12)
        ax.set_xlabel('training steps', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel(f'smoothed reward (window={REWARD_SMOOTHING_WINDOW})', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
    
    plt.suptitle("per-client type comparison", fontsize=16)
    plt.tight_layout()
    
    # save
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")
    plt.close()


def generate_all_plots(logs_dir="logs"):
    """
    automatically detects and plots all experiments in logs/ directory.
    
    args:
        logs_dir: path to logs directory
    """
    print(f"\n{'='*60}")
    print(f"generating plots from: {logs_dir}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(logs_dir):
        print(f"error: {logs_dir} directory not found")
        return
    
    # find all experiment directories
    experiment_dirs = [d for d in os.listdir(logs_dir) 
                      if os.path.isdir(os.path.join(logs_dir, d))]
    
    if not experiment_dirs:
        print(f"no experiment directories found in {logs_dir}")
        return
    
    print(f"found {len(experiment_dirs)} experiments:")
    for exp in experiment_dirs:
        print(f"  - {exp}")
    print()
    
    # generate individual plots for each experiment
    for exp_name in experiment_dirs:
        exp_path = os.path.join(logs_dir, exp_name)
        data = load_experiment_data(exp_path)
        
        if not data.empty:
            plot_learning_curves(
                data,
                output_filename=f"{exp_name}_learning_curves.png",
                title=f"{exp_name} - learning curves"
            )
    
    # generate comparison plots
    experiments_dict = {
        exp: os.path.join(logs_dir, exp) 
        for exp in experiment_dirs
    }
    
    if len(experiments_dict) > 1:
        plot_comparison(
            experiments_dict,
            output_filename="all_experiments_comparison.png",
            title="all experiments comparison"
        )
        
        plot_per_client_comparison(
            experiments_dict,
            output_filename="per_client_comparison.png"
        )
    
    print(f"\n{'='*60}")
    print(f"all plots saved to: {PLOTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # generate all plots from logs directory
    generate_all_plots("logs")
