#!/usr/bin/env python3
"""
generate plots from moonlanderv4 experiment results.

usage:
    python generate_plots.py                    # plot all experiments in logs/
    python generate_plots.py logs/fl_run        # plot specific experiment
"""

import sys
from plotting import generate_all_plots, load_experiment_data, plot_learning_curves

def main():
    if len(sys.argv) > 1:
        # plot specific experiment
        exp_path = sys.argv[1]
        print(f"plotting experiment: {exp_path}")
        
        data = load_experiment_data(exp_path)
        if not data.empty:
            exp_name = exp_path.split('/')[-1]
            plot_learning_curves(
                data,
                output_filename=f"{exp_name}_learning_curves.png",
                title=f"{exp_name} - learning curves"
            )
        else:
            print("no data found!")
    else:
        # plot all experiments
        generate_all_plots("logs")

if __name__ == "__main__":
    main()
