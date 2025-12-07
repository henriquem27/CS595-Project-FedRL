import torch as th
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from sklearn.cluster import KMeans
import os
import csv
import json
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander


class ExperimentLogger:
    def __init__(self, experiment_name):
        self.base_dir = f"logs/{experiment_name}"
        self.weights_dir = os.path.join(self.base_dir, "weights")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        
        # Create directories
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Cache for CSV writers to avoid reopening files constantly
        self.csv_headers_written = set()

    def log_metric(self, client_label, metric_name, value, step, round_num):
        """
        Logs a scalar metric (reward, length, etc.) to a CSV file specific to that client.
        """
        filename = os.path.join(self.metrics_dir, f"{client_label}_metrics.csv")
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists and filename not in self.csv_headers_written:
                writer.writerow(['round', 'step', 'metric', 'value'])
                self.csv_headers_written.add(filename)
            
            writer.writerow([round_num, step, metric_name, value])

    def save_client_weights(self, client_label, round_num, state_dict):
        """
        Saves model weights to a specific folder structure: weights/round_X/client_Y.pt
        """
        # Create a folder specifically for this round
        round_dir = os.path.join(self.weights_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        save_path = os.path.join(round_dir, f"{client_label}.pt")
        
        # Option 1: Save full PyTorch dict (Easiest to reload)
        th.save(state_dict, save_path)
        
        # Option 2: Save as numpy (If you strictly want arrays like your current setup)
        # flat_weights = np.concatenate([p.cpu().numpy().flatten() for p in state_dict.values()])
        # np.savez_compressed(save_path.replace('.pt', '.npz'), weights=flat_weights)
class StreamingCallback(BaseCallback):
    """
    A callback that logs rewards to disk immediately after an episode ends,
    rather than storing them in RAM.
    """
    def __init__(self, logger, agent_label, round_num, verbose=0):
        super(StreamingCallback, self).__init__(verbose)
        # RENAMED: self.logger -> self.exp_logger to avoid conflict with SB3 BaseCallback
        self.exp_logger = logger 
        self.agent_label = agent_label
        self.round_num = round_num

    def _on_step(self) -> bool:
        # Check for episode completion
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                # Extract reward and length from the monitor info
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    
                    # Log to disk immediately using the RENAMED variable
                    self.exp_logger.log_metric(
                        client_label=self.agent_label,
                        metric_name="episode_reward",
                        value=ep_reward,
                        step=self.n_calls,
                        round_num=self.round_num
                    )
                    self.exp_logger.log_metric(
                        client_label=self.agent_label,
                        metric_name="episode_length",
                        value=ep_length,
                        step=self.n_calls,
                        round_num=self.round_num
                    )
        return True

class DynamicLunarLander(LunarLander):
    """
    A subclass of LunarLander that allows changing gravity/wind 
    without destroying the Python process.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reconfigure(self, gravity, wind_power):
        """
        Updates physics parameters and resets the Box2D world.
        """
        # Update internal parameters
        self.gravity = gravity
        self.enable_wind = True
        self.wind_power = wind_power
        
        # Force a world reset (Box2D specific)
        if self.world is not None:
            # We don't destroy the world object directly here as reset() handles it
            # but we need to ensure the next reset() uses new values.
            pass
        
        # The next time reset() is called by the agent, 
        # LunarLander._destroy() and _create_particle() uses self.gravity
        return True

def _flatten_model_weights(state_dicts):
    """
    Converts a list of model state_dicts into a 2D numpy array
    (n_models, n_parameters) for clustering.
    """
    flattened_weights = []
    for state_dict in state_dicts:
        model_vector = []
        for key in state_dict:
            # Move tensor to CPU, convert to numpy, and flatten
            tensor_flat = state_dict[key].cpu().numpy().ravel()
            model_vector.append(tensor_flat)

        # Concatenate all layer vectors into one big vector for this model
        if model_vector:
            flattened_weights.append(np.concatenate(model_vector))

    return np.array(flattened_weights)


def find_closest_model(client_state_dict, global_state_dicts):
    """
    Finds which global model (by index) is "closest" to the client's
    current model using L2 distance.
    """
    min_dist = np.inf
    best_idx = 0

    # Ensure client dict is on CPU for comparison
    client_dict_cpu = {k: v.cpu() for k, v in client_state_dict.items()}

    for idx, global_dict in enumerate(global_state_dicts):
        dist = 0
        # Calculate L2 distance (sum of squared differences)
        for key in client_dict_cpu:
            dist += th.sum((client_dict_cpu[key] -
                           global_dict[key].cpu())**2).item()

        if dist < min_dist:
            min_dist = dist
            best_idx = idx

    return best_idx


def cluster_and_average_models(state_dicts, n_clusters):
    """
    Performs K-Means clustering on a list of model state_dicts
    and returns a list of new, averaged state_dicts (one per cluster).
    
    Returns:
        list[OrderedDict]: A list of averaged models.
    """
    if not state_dicts:
        return []

    # 1. Convert state_dicts to a 2D array for clustering
    weight_matrix = _flatten_model_weights(state_dicts)

    if len(weight_matrix) < n_clusters:
        print(
            f"  Warning: Not enough models ({len(weight_matrix)}) for {n_clusters} clusters.")
        print("  Falling back to standard FedAvg.")
        # Fallback: Just average all models into one
        return [average_ordered_dicts(state_dicts)]

    # 2. Run K-Means clustering
    print(
        f"  Clustering {len(weight_matrix)} models into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weight_matrix)

    # 3. Average models within each cluster
    averaged_models = []
    for i in range(n_clusters):
        # Get the indices of the models belonging to this cluster
        indices_in_cluster = np.where(cluster_labels == i)[0]

        if len(indices_in_cluster) == 0:
            print(f"  Warning: Cluster {i} is empty. Skipping.")
            continue

        # Get the state_dicts for those models
        models_in_cluster = [state_dicts[idx] for idx in indices_in_cluster]

        # Use your existing averaging function on this subset
        averaged_cluster_model = average_ordered_dicts(models_in_cluster)
        averaged_models.append(averaged_cluster_model)
        print(f"  Cluster {i} (size {len(indices_in_cluster)}) averaged.")

    return averaged_models
def average_state_dicts(state_dicts: list[OrderedDict[str, th.Tensor]]) -> OrderedDict[str, th.Tensor]:
    """
    Averages a list of PyTorch state dictionaries.
    This is the core of Federated Averaging (FedAvg).
    """
    if not state_dicts:
        return None

    num_clients = len(state_dicts)
    avg_state_dict = OrderedDict()

    # Get all parameter keys from the first client
    for key in state_dicts[0].keys():
        # Start with the first client's tensor and clone it
        sum_tensor = state_dicts[0][key].clone().float()

        # Add the tensors from all other clients
        for i in range(1, num_clients):
            sum_tensor += state_dicts[i][key]

        # Average the sum
        avg_state_dict[key] = sum_tensor / num_clients

    return avg_state_dict

def average_ordered_dicts(dict_list: list[OrderedDict]) -> OrderedDict:
    """
    Averages a list of PyTorch state_dicts or delta dicts.
    """
    if not dict_list:
        return OrderedDict()

    avg_dict = OrderedDict()
    # Get keys from the first dictionary
    for key in dict_list[0].keys():
        # Sum up all tensors for this key
        avg_tensor = sum(d[key] for d in dict_list)
        # Divide by the number of clients
        avg_tensor = avg_tensor / len(dict_list)
        avg_dict[key] = avg_tensor
    return avg_dict

class WeightStorageCallback(BaseCallback):
    """
    A custom callback to store model weights AND episode rewards during training.
    """

    def __init__(self, check_freq: int, agent_label: str, verbose: int = 0):
        super(WeightStorageCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.agent_label = agent_label
        self.current_epoch = 0
        # --- Data lists for weights ---
        self.weights_log = []
        self.labels_log = []
        self.steps_log = []
        self.epoch_log = []

        # --- Data lists for episode rewards ---
        self.ep_rewards_log = []
        self.ep_lengths_log = []
        self.ep_labels_log = []
        self.ep_steps_log = []  # Step count when episode ended
        self.ep_epoch_log = []  # Epoch count when episode ended
    def update_epoch(self, epoch_num: int):
        self.current_epoch = epoch_num
    def _on_step(self) -> bool:
        """
        This method is called after each environment step.
        """

        # --- 1. Log weights at the specified frequency ---
        if self.n_calls % self.check_freq == 0:
            # Get the model's state dictionary
            state_dict = self.model.policy.state_dict()

            # Flatten all parameters into a single 1D numpy vector
            flat_weights = np.concatenate([
                param.cpu().detach().numpy().flatten()
                for param in state_dict.values()
            ])

            # Store the weight data
            self.weights_log.append(flat_weights)
            self.labels_log.append(self.agent_label)
            self.steps_log.append(self.n_calls)
            self.epoch_log.append(self.current_epoch)

            if self.verbose > 0:
                print(
                    f"Step {self.n_calls}: Stored weights for {self.agent_label} (size: {flat_weights.shape[0]})")

        # --- 2. Log episode info on completion ---
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]

                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]

                    # Store this episode's data
                    self.ep_rewards_log.append(ep_reward)
                    self.ep_lengths_log.append(ep_length)
                    self.ep_labels_log.append(self.agent_label)
                    self.ep_steps_log.append(self.n_calls)
                    self.ep_epoch_log.append(self.current_epoch)
                    if self.verbose > 1:
                        print(
                            f"Step {self.n_calls}: Logged episode for {self.agent_label} (Reward: {ep_reward}, Length: {ep_length})")

        # Return True to continue training
        return True



def save_data(client_callbacks, output_filename):
    all_weights = []
    all_weight_labels = []
    all_weight_steps = []
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_labels = []
    all_ep_steps = []
    all_weight_epochs = []
    all_ep_epochs = []


    for cb in client_callbacks:
        all_weights.extend(cb.weights_log)
        all_weight_labels.extend(cb.labels_log)
        all_weight_steps.extend(cb.steps_log)
        all_ep_rewards.extend(cb.ep_rewards_log)
        all_ep_lengths.extend(cb.ep_lengths_log)
        all_ep_labels.extend(cb.ep_labels_log)
        all_ep_steps.extend(cb.ep_steps_log)
        all_ep_epochs.extend(cb.ep_epoch_log)
    # --- Save all arrays to a single compressed file ---
    np.savez_compressed(
        output_filename,
        weights=np.array(all_weights),
        weight_labels=np.array(all_weight_labels),
        weight_steps=np.array(all_weight_steps),
        weight_epochs=np.array(all_weight_epochs),
        ep_rewards=np.array(all_ep_rewards),
        ep_lengths=np.array(all_ep_lengths),
        ep_labels=np.array(all_ep_labels),
        ep_steps=np.array(all_ep_steps),
        ep_epochs=np.array(all_ep_epochs)
    )

    print(
        f"Successfully saved all federated training data to {output_filename}")


def average_deltas(deltas: list[OrderedDict[str, th.Tensor]]) -> OrderedDict[str, th.Tensor]:
    """
    Averages a list of PyTorch state dictionaries (which are deltas).
    (This is your original `average_state_dicts` function, just renamed
    for clarity, as it's now averaging updates, not full states)
    """
    if not deltas:
        return None

    num_clients = len(deltas)
    avg_delta = OrderedDict()

    # Get all parameter keys from the first client
    for key in deltas[0].keys():
        # Start with the first client's tensor and clone it
        sum_tensor = deltas[0][key].clone().float()

        # Add the tensors from all other clients
        for i in range(1, num_clients):
            sum_tensor += deltas[i][key]

        # Average the sum
        avg_delta[key] = sum_tensor / num_clients

    return avg_delta
