import numpy as np
from plotting import plot_learning_curves, plot_reward_vs_epoch, plot_tsne_weights, plot_umap_weights
from comparison_plot import plot_combined_learning_curves

def generate_plots_fl(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png")

def generate_plots_single(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png")
def generate_plots_dp(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png")
if __name__ == "__main__":
    print("Loading data from 'federated_training_data.npz'...")
    data = np.load('federated_training_data.npz', allow_pickle=True)

    print("Generating plots...")
    generate_plots_fl(data, "fl")  # Output files will be prefixed with "fl"
    print("All plots generated and saved.")

    data = np.load('training_data.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots_single(data, "single")  # Output files will be prefixed with "single"
    print("All plots generated and saved.") 

    data = np.load('dp_training_data.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots_dp(data, "dp")  # Output files will be prefixed with "dp"
    print("All plots generated and saved.") 
    
    plot_combined_learning_curves()
    print("Combined learning curve plot generated and saved.")