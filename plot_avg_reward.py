import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_avg_reward():
    # Load data
    file_path = 'sv_results/v4/all_metrics_pivoted.csv'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    df_pivoted = pd.read_csv(file_path)

    # Sort data to ensure correct episode ordering
    df_pivoted = df_pivoted.sort_values(['run_type', 'client_id', 'step'])

    # Create episode_num per client
    df_pivoted['episode_num'] = df_pivoted.groupby(['run_type', 'client_id']).cumcount()

    # Calculate average reward
    avg_reward = df_pivoted.groupby(['run_type', 'episode_num'])['episode_reward'].mean().reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_reward, x='episode_num', y='episode_reward', hue='run_type')
    plt.title('Average Reward by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    # Save plot to verify
    output_path = 'avg_reward_plot_fixed.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_avg_reward()
