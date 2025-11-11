import torch as th
import gymnasium as gym
import gymnasium_robotics 
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict


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


class PartialObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to mask (zero out) parts of the observation space.
    """

    def __init__(self, env, mask_indices):
        super(PartialObservationWrapper, self).__init__(env)
        self.mask_indices = mask_indices
        # The observation_space shape itself does NOT change

    def _mask_obs(self, obs):
        """Masks a single observation."""
        masked_obs = obs.copy()
        obs_shape = obs.shape
        obs_flat = masked_obs.flatten()

        # Set specified indices to zero
        obs_flat[self.mask_indices] = 0.0

        return obs_flat.reshape(obs_shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._mask_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._mask_obs(obs), info


class WeightStorageCallback(BaseCallback):
    """
    A custom callback to store model weights AND episode rewards during training.
    """

    def __init__(self, check_freq: int, agent_label: str, verbose: int = 0):
        super(WeightStorageCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.agent_label = agent_label

        # --- Data lists for weights ---
        self.weights_log = []
        self.labels_log = []
        self.steps_log = []

        # --- Data lists for episode rewards ---
        self.ep_rewards_log = []
        self.ep_lengths_log = []
        self.ep_labels_log = []
        self.ep_steps_log = []  # Step count when episode ended

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

                    if self.verbose > 1:
                        print(
                            f"Step {self.n_calls}: Logged episode for {self.agent_label} (Reward: {ep_reward}, Length: {ep_length})")

        # Return True to continue training
        return True


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
def run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, DP_SENSITIVITY, DP_EPSILON):
    """
    Runs a federated learning experiment with differential privacy.

    Args:
        NUM_ROUNDS (int): Number of federation rounds.
        CHECK_FREQ (int): Frequency for callbacks to log data (in steps).
        LOCAL_STEPS (int): Number of training steps per client per round.
        task_list (list of dicts): Defines the clients.
        DP_SENSITIVITY (float): L1 sensitivity (clipping bound).
        DP_EPSILON (float): Privacy budget (epsilon).
    """

    # --- Differential Privacy Setup ---
    # The scale 'b' for Laplace noise is Sensitivity / Epsilon

    DP_SCALE = DP_SENSITIVITY / DP_EPSILON
    print(
        f"DP settings: Epsilon={DP_EPSILON}, Sensitivity={DP_SENSITIVITY}, Noise Scale={DP_SCALE:.4f}")

    # === Model Initialization ===

    # 1. Create the Global Model (Server Model)
    dummy_env = gym.make('InvertedDoublePendulum-v5')
    global_model = PPO("MlpPolicy", dummy_env, verbose=0)
    dummy_env.close()

    # 2. Create Client Models, Envs, and Callbacks dynamically
    client_models = []
    client_envs = []
    client_callbacks = []

    print("Initializing clients...")
    for i, task in enumerate(task_list):
        label = task['label']
        mask = task['mask']
        print(f"  > Client {i+1} ({label}): Mask={mask if mask else 'None'}")

        # Create Env
        env_base = gym.make('InvertedDoublePendulum-v5')
        if mask is not None:
            env = PartialObservationWrapper(env_base, mask_indices=mask)
        else:
            env = env_base
        client_envs.append(env)

        # Create Client Model
        client = PPO("MlpPolicy", env, verbose=0)
        client_models.append(client)

        # Create Callback
        callback = WeightStorageCallback(
            check_freq=CHECK_FREQ,
            agent_label=label
        )
        client_callbacks.append(callback)

    print(
        f"\nStarting DP Federated Learning: {len(task_list)} clients, {NUM_ROUNDS} rounds, {LOCAL_STEPS} local steps per round.")

    # === Federated Training Loop ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        # 1. Broadcast global model weights to clients
        # Keep a copy of the *original* global weights to calculate deltas
        global_state_dict = global_model.policy.state_dict()

        for client in client_models:
            client.policy.load_state_dict(global_state_dict)

        # 2. Local Training
        for i, (client, callback) in enumerate(zip(client_models, client_callbacks)):
            print(f"Training {task_list[i]['label']}...")
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

        # -----------------------------------------------------------------
        # 3. COMPUTE, CLIP, AND NOISE DELTAS
        # -----------------------------------------------------------------
        print("Clipping and noising client deltas...")
        noisy_deltas = []

        for i, client in enumerate(client_models):
            # Get the client's new weights
            new_weights = client.policy.state_dict()

            delta = OrderedDict()
            noisy_delta = OrderedDict()
            total_l1_norm = 0.0

            # --- First pass: Calculate delta and its total L1 norm ---
            for key in global_state_dict.keys():
                delta[key] = new_weights[key] - global_state_dict[key]
                total_l1_norm += th.sum(th.abs(delta[key]))

            total_l1_norm = total_l1_norm.item()
            print(
                f"    - {task_list[i]['label']} L1 norm: {total_l1_norm:.4f}")

            # --- Calculate the clipping factor ---
            # Clip factor = min(1, S / ||delta||_1)
            clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))

            # --- Second pass: Apply clipping and add Laplace noise ---
            for key in delta.keys():
                clipped_delta = delta[key] * clip_factor

                noise = th.tensor(
                    np.random.laplace(0, scale=DP_SCALE,
                                      size=clipped_delta.shape),
                    dtype=clipped_delta.dtype,
                    device=clipped_delta.device
                )
                noisy_delta[key] = clipped_delta + noise

            noisy_deltas.append(noisy_delta)

        # -----------------------------------------------------------------
        # 4. AGGREGATE and UPDATE
        # -----------------------------------------------------------------
        print("Aggregating noisy deltas...")
        avg_noisy_delta = average_ordered_dicts(noisy_deltas)

        # Update global model by *adding* the average noisy delta
        new_global_state_dict = OrderedDict()
        for key in global_state_dict.keys():
            new_global_state_dict[key] = global_state_dict[key] + \
                avg_noisy_delta[key]

        global_model.policy.load_state_dict(new_global_state_dict)

    print("\n--- Federated Learning Complete ---")

    # --- Data Collection and Saving (Dynamic) ---
    print("Collecting data from callbacks...")

    all_weights = []
    all_weight_labels = []
    all_weight_steps = []
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_labels = []
    all_ep_steps = []

    for cb in client_callbacks:
        all_weights.extend(cb.weights_log)
        all_weight_labels.extend(cb.labels_log)
        all_weight_steps.extend(cb.steps_log)
        all_ep_rewards.extend(cb.ep_rewards_log)
        all_ep_lengths.extend(cb.ep_lengths_log)
        all_ep_labels.extend(cb.ep_labels_log)
        all_ep_steps.extend(cb.ep_steps_log)

    # --- Save all arrays to a single compressed file ---
    output_filename = 'dif_federated_training_data_dp.npz'
    np.savez_compressed(
        output_filename,
        weights=np.array(all_weights),
        weight_labels=np.array(all_weight_labels),
        weight_steps=np.array(all_weight_steps),
        ep_rewards=np.array(all_ep_rewards),
        ep_lengths=np.array(all_ep_lengths),
        ep_labels=np.array(all_ep_labels),
        ep_steps=np.array(all_ep_steps)
    )

    print(
        f"Successfully saved all federated training data to {output_filename}")

    # Clean up all environments
    print("Closing environments...")
    for env in client_envs:
        env.close()


# ==================================================================
#                  EXAMPLE USAGE
# ==================================================================

if __name__ == '__main__':

    print("Running dp_fl_training_utils.py as main script...")

    # === Define FL Hyperparameters ===
    NUM_ROUNDS = 20
    LOCAL_STEPS = 5000
    CHECK_FREQ = 5000  # Callback check frequency

    # === Define DP Hyperparameters ===
    DP_SENSITIVITY = 150.0  # L1 clipping norm (S)
    DP_EPSILON = 300.0      # Privacy budget

    # === Define the Clients (Tasks) ===
    # This list now drives the entire experiment
    fl_task_list = [
        {
            'label': 'Client_1_Standard',
            'mask': None
        },
        {
            'label': 'Client_2_Masked_67',
            'mask': [6, 7]
        },
        {
            'label': 'Client_3_Masked_234',
            'mask': [2, 3, 4]
        }
    ]

    # === Run the Experiment ===
    # (This was the missing piece in your original file)
    run_dp_fl_experiment(
        NUM_ROUNDS,
        CHECK_FREQ,
        LOCAL_STEPS,
        fl_task_list,
        DP_SENSITIVITY,
        DP_EPSILON
    )
