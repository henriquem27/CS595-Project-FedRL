import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict

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

class PartialObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to mask (zero out) parts of the observation space.
    
    :param env: The environment to wrap
    :param mask_indices: A list of indices in the flat observation
                         to set to zero.
    """

    def __init__(self, env, mask_indices):
        super(PartialObservationWrapper, self).__init__(env)
        self.mask_indices = mask_indices
        # The observation_space shape itself does NOT change

    def _mask_obs(self, obs):
        """Masks a single observation."""
        # Make a copy to avoid modifying the original
        masked_obs = obs.copy()

        # Flatten to easily apply indices
        # (For pendulum envs, obs is already flat, but this is safer)
        obs_shape = obs.shape
        obs_flat = masked_obs.flatten()

        # Set specified indices to zero
        obs_flat[self.mask_indices] = 0.0

        # Reshape back to the original observation shape
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

        # --- NEW: Data lists for episode rewards ---
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

        # --- 2. NEW: Log episode info on completion ---
        # PPO runs in vectorized environments (even if n_envs=1),
        # so we must check all 'dones' flags.
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # An episode has just finished in the i-th environment
                info = self.locals.get("infos", [{}])[i]

                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]

                    # Store this episode's data
                    self.ep_rewards_log.append(ep_reward)
                    self.ep_lengths_log.append(ep_length)
                    self.ep_labels_log.append(self.agent_label)
                    self.ep_steps_log.append(self.n_calls)

                    if self.verbose > 1:  # Set verbose=2 in callback to see this
                        print(
                            f"Step {self.n_calls}: Logged episode for {self.agent_label} (Reward: {ep_reward}, Length: {ep_length})")

        # Return True to continue training
        return True


def run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list):
    """
    Runs a federated learning experiment with a dynamic number of clients.

    Args:
        NUM_ROUNDS (int): Number of federation rounds.
        CHECK_FREQ (int): Frequency for callbacks to log data (in steps).
        LOCAL_STEPS (int): Number of training steps per client per round.
        task_list (list of dicts): Defines the clients.
            Example: [{'label': 'Client_1', 'mask': None},
                      {'label': 'Client_2', 'mask': [6, 7]}]
    """

    # === Model Initialization ===

    # 1. Create the Global Model (Server Model)
    # It needs a dummy env just to initialize the network architecture
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
        f"\nStarting Federated Learning: {len(task_list)} clients, {NUM_ROUNDS} rounds, {LOCAL_STEPS} local steps per round.")

    # === Federated Training Loop ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        # 1. Broadcast global model weights to all clients
        global_state_dict = global_model.policy.state_dict()
        for client in client_models:
            client.policy.load_state_dict(global_state_dict)

        # 2. Local Training (loop over all clients)
        for i, (client, callback) in enumerate(zip(client_models, client_callbacks)):
            print(f"Training {task_list[i]['label']}...")
            # reset_num_timesteps=False is CRUCIAL for FL.
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

        # 3. Aggregation (FedAvg)
        print("Aggregating model weights...")
        client_state_dicts = [
            client.policy.state_dict() for client in client_models
        ]

        avg_state_dict = average_state_dicts(client_state_dicts)

        # 4. Update global model
        global_model.policy.load_state_dict(avg_state_dict)

    print("\n--- Federated Learning Complete ---")

    # --- Data Collection and Saving (Dynamic) ---

    print("Collecting data from callbacks...")

    # Create lists to hold aggregated data
    all_weights = []
    all_weight_labels = []
    all_weight_steps = []
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_labels = []
    all_ep_steps = []

    # Loop through all callbacks and extend the master lists
    for cb in client_callbacks:
        all_weights.extend(cb.weights_log)
        all_weight_labels.extend(cb.labels_log)
        all_weight_steps.extend(cb.steps_log)

        all_ep_rewards.extend(cb.ep_rewards_log)
        all_ep_lengths.extend(cb.ep_lengths_log)
        all_ep_labels.extend(cb.ep_labels_log)
        all_ep_steps.extend(cb.ep_steps_log)

    # --- Save all arrays to a single compressed file ---
    output_filename = 'federated_training_data.npz'
    np.savez_compressed(
        output_filename,

        # Weight data
        weights=np.array(all_weights),
        weight_labels=np.array(all_weight_labels),
        weight_steps=np.array(all_weight_steps),

        # Episode data
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



# This block only runs when you execute `python fl_training_utils.py`
if __name__ == '__main__':

    print("Running fl_training_utils.py as main script...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 5
    LOCAL_STEPS = 10000  # Steps *per round*
    CHECK_FREQ = 2000    # Log weights every 2000 steps

    # 2. Define the clients
    # This single list defines all your clients.
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
        },
        # You can add more clients just by editing this list!
        # {
        #     'label': 'Client_4_Standard_B',
        #     'mask': None
        # }
    ]

    # 3. Run the experiment
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, fl_task_list)
