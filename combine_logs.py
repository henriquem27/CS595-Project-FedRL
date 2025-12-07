import os
import pandas as pd
import glob

def combine_logs(base_path, output_file):
    all_metrics = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        if 'metrics' in dirs:
            metrics_path = os.path.join(root, 'metrics')
            run_type = os.path.basename(root) # e.g., dp_fl_sens1.0_eps30.0 or fl_run
            
            # Iterate over CSV files in the metrics directory
            for filename in os.listdir(metrics_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(metrics_path, filename)
                    
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Extract client info from filename
                        # Expected format: Client_ID_Planet_metrics.csv or similar
                        # Example: Client_0_Earth_metrics.csv
                        client_id_str = filename.replace('_metrics.csv', '')
                        
                        # Add metadata columns
                        df['run_type'] = run_type
                        df['client_id'] = client_id_str
                        
                        all_metrics.append(df)
                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined {len(all_metrics)} files into {output_file}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No metrics files found.")

if __name__ == "__main__":
    base_path = 'sv_results/v4/logs'
    output_file = 'sv_results/v4/all_metrics.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combine_logs(base_path, output_file)
