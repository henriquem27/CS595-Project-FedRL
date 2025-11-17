import numpy as np
from plotting import plot_learning_curves, plot_reward_vs_epoch, plot_tsne_weights, plot_umap_weights
from comparison_plot import plot_combined_learning_curves

def generate_plots_fl(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png",title="Federated Learning")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png",title="Federated Learning")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png",  title="Federated Learning")

def generate_plots_single(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png",title="Single Client Training")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png",title="Single Client Training")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png",title="Single Client Training")
def generate_plots_dp(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png",title="Differentially Private FL")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png",title="Differentially Private FL")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png",title="Differentially Private FL")

def generate_plots_clustered(data, output_prefix):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png",title="Clustered FL")
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png",    title="Clustered FL")
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png",    title="Clustered FL")
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

    data = np.load('clustered_fl_training_data.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots_clustered(data, "cluster_fl")  # Output files will be prefixed with "dp"
    print("All plots generated and saved.")
    
    # plot_combined_learning_curves()
    # print("Combined learning curve plot generated and saved.")