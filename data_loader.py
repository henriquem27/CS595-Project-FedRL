import re
import torch
import os
import numpy as np
import glob



def load_data_recursively(base_dir, max_rounds=None):

    weights_list = []
    labels_list = []
    steps_list = []
    
    print(f"Scanning {base_dir}...")
    
    # Pattern to extract round number, e.g., "round_15" -> 15
    round_pattern = re.compile(r'round_(\d+)')
    
    # Walk through directory
    for root, dirs, files in os.walk(base_dir):
        dir_name = os.path.basename(root)
        match = round_pattern.match(dir_name)
        
        if match:
            round_num = int(match.group(1))
            
            # Filter by max_rounds if specified
            if max_rounds is not None and round_num > max_rounds:
                continue
                
            for file in files:
                if file.endswith(".pt"):
                    if "Global_Model" in file:
                        continue
                        
                    file_path = os.path.join(root, file)
                    # Label is filename without extension, e.g., "Moon_UK_1"
                    label = file.replace('.pt', '')
                    
                    try:
                        # Load PyTorch checkpoint
                        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
                        
                        # Flatten weights into single vector
                        flat_w = []
                        for key in sorted(state_dict.keys()):
                            param = state_dict[key]
                            if isinstance(param, torch.Tensor):
                                # Flatten and move to cpu numpy
                                flat_w.append(param.cpu().view(-1).numpy())
                        
                        if flat_w:
                            # Concatenate all flattened layers
                            weight_vector = np.concatenate(flat_w)
                            
                            weights_list.append(weight_vector)
                            labels_list.append(label)
                            steps_list.append(round_num)
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    # Process results into numpy format expected by plotting functions
    if len(weights_list) > 0:
        data = {
            'weights': np.stack(weights_list),
            'weight_labels': np.array(labels_list),
            'weight_steps': np.array(steps_list)
        }
        print(f"Loaded {len(weights_list)} weight vectors.")
        return data
    else:
        print("No weights found.")
        return {'weights': [], 'weight_labels': [], 'weight_steps': []}
