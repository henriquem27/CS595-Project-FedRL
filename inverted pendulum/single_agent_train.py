import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO


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


def run_experiment_training(TOTAL_TIMESTEPS, CHECK_FREQ, task_list):
    """
    Trains multiple agents based on a list of task configurations.

    Args:
        TOTAL_TIMESTEPS (int): Total timesteps to train each agent.
        CHECK_FREQ (int): Frequency to save weights in the callback.
        task_list (list of dicts): A list where each dict defines an agent.
            Example: [
                {'label': 'Standard', 'mask': None},
                {'label': 'Masked67', 'mask': [6, 7]},
                {'label': 'Masked34', 'mask': [3, 4]}
            ]
    """

    # --- Lists to aggregate data from all agents ---
    all_weights_logs = []
    all_weight_labels_logs = []
    all_weight_steps_logs = []

    all_ep_rewards_logs = []
    all_ep_lengths_logs = []
    all_ep_labels_logs = []
    all_ep_steps_logs = []

    # --- Loop over all defined tasks ---
    for i, task in enumerate(task_list):
        agent_num = i + 1
        indices_to_mask = task['mask']
        agent_label = task['label']

        print(f"--- Training Agent {agent_num} ({agent_label}) ---")

        # 1. Create the environment
        env_base = gym.make('InvertedDoublePendulum-v5')

        if indices_to_mask is not None:
            # Apply the wrapper to alter the task
            print(f"Applying observation mask for indices: {indices_to_mask}")
            env = PartialObservationWrapper(
                env_base, mask_indices=indices_to_mask)
        else:
            # Use the standard, unwrapped environment
            print("Using standard (unmasked) environment.")
            env = env_base

        # 2. Create the model and callback
        model = PPO("MlpPolicy", env, verbose=0)

        callback = WeightStorageCallback(
            check_freq=CHECK_FREQ,
            agent_label=agent_label
        )

        # 3. Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        print(f"Agent {agent_num} ({agent_label}) training complete.")

        # 4. Store the results from this agent
        all_weights_logs.extend(callback.weights_log)
        all_weight_labels_logs.extend(callback.labels_log)
        all_weight_steps_logs.extend(callback.steps_log)

        all_ep_rewards_logs.extend(callback.ep_rewards_log)
        all_ep_lengths_logs.extend(callback.ep_lengths_log)
        all_ep_labels_logs.extend(callback.ep_labels_log)
        all_ep_steps_logs.extend(callback.ep_steps_log)

        env.close()  # Good practice to close envs

    # --- Aggregate all data into numpy arrays ---
    print("\nAggregating data from all agents...")

    all_weights = np.array(all_weights_logs)
    all_weight_labels = np.array(all_weight_labels_logs)
    all_weight_steps = np.array(all_weight_steps_logs)

    all_ep_rewards = np.array(all_ep_rewards_logs)
    all_ep_lengths = np.array(all_ep_lengths_logs)
    all_ep_labels = np.array(all_ep_labels_logs)
    all_ep_steps = np.array(all_ep_steps_logs)

    # --- Save all arrays to a single compressed file ---
    output_filename = 'training_data.npz'
    np.savez_compressed(
        output_filename,

        # Weight data
        weights=all_weights,
        weight_labels=all_weight_labels,
        weight_steps=all_weight_steps,

        # Episode data
        ep_rewards=all_ep_rewards,
        ep_lengths=all_ep_lengths,
        ep_labels=all_ep_labels,
        ep_steps=all_ep_steps
    )

    print(f"Successfully saved all training data to {output_filename}")

if __name__ == "__main__":
    # Example usage
    tasks_to_run = [
        {
            'label': 'StandardDoublePendulum',
            'mask': None  # This is your "Agent 1"
        },
        {
            'label': 'PenalizedDoublePendulum67',
            'mask': [6, 7]  # This is your "Agent 2"
        },
        {
            'label': 'PenalizedDoublePendulum34',
            'mask': [3, 4]  # This is your "Agent 3"
        },
        # You can easily add more agents here!
        # {
        #     'label': 'PenalizedDoublePendulum123',
        #     'mask': [1, 2, 3]
        # }
    ]

    run_experiment_training(
        TOTAL_TIMESTEPS=100000,
        CHECK_FREQ=1000,
        task_list=tasks_to_run
    )       