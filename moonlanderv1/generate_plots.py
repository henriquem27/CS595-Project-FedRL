import numpy as np
from plotting import plot_learning_curves, plot_tsne_weights, plot_umap_weights
from comparison_plot import plot_combined_learning_curves

def generate_plots(data, output_prefix,title):
    # Generate all plots
    plot_learning_curves(data, f"{output_prefix}_learning_curves_epochs.png",title=title)
    plot_tsne_weights(data, f"{output_prefix}_tsne_weights_markers.png",title=title)
    plot_umap_weights(data, f"{output_prefix}_umap_weights_markers.png",  title=title)

if __name__ == "__main__":
    print("Loading data from 'federated_training_data.npz'...")
    data = np.load('federated_training_data.npz', allow_pickle=True)

    print("Generating plots...")
    generate_plots(data, "fl", "Federated Learning")  # Output files will be prefixed with "fl"
    print("All plots generated and saved.")

    data = np.load('training_data.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots(data, "single", "Single Client Training")  # Output files will be prefixed with "single"
    print("All plots generated and saved.") 


    data = np.load('dp_training_data_ep5_sens15.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots(data, "dp_ep5_sens15", "Differentially Private FL epsilon=5, sensitivity=15")  # Output files will be prefixed with "dp"
    print("All plots generated and saved.")

    data = np.load('dp_training_data_ep10_sens15.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots(data, "dp_ep10_sens15", "Differentially Private FL epsilon=10, sensitivity=15")  # Output files will be prefixed with "dp"
    print("All plots generated and saved.")

    data = np.load('dp_training_data_ep30_sens15.npz', allow_pickle=True)
    print("Generating plots...")
    generate_plots(data, "dp_ep30_sens15", "Differentially Private FL epsilon=30, sensitivity=15")  # Output files will be prefixed with "dp"
    print("All plots generated and saved.")
