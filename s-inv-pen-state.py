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


def main():
    TOTAL_TIMESTEPS = 100000
    CHECK_FREQ = 5000

    # === Agent 1 (Standard Task) ===
    # Use the double pendulum environment
    env_1 = gym.make('InvertedDoublePendulum-v5')
    model_1 = PPO("MlpPolicy", env_1, verbose=0)

    callback_1 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='StandardDoublePendulum'  # Updated label
    )

    print("Training Agent 1 (StandardDoublePendulum)...")
    model_1.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_1)
    print("Agent 1 training complete.")

    # === Agent 2 (Altered Task) ===
    # Use the *same* base environment
    env_2_base = gym.make('InvertedDoublePendulum-v5')
    # Apply the wrapper to alter its task
    indices_to_mask = [6,7]
    env_2 = PartialObservationWrapper(env_2_base, mask_indices=indices_to_mask)

    model_2 = PPO("MlpPolicy", env_2, verbose=0)

    callback_2 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='PenalizedDoublePendulum67'  # Updated label
    )

    print("Training Agent 3 (PenalizedDoublePendulum)...")
    model_2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_2)
    print("Agent 3 training complete.")

       # === Agent 3 (Altered Task) ===
    # Use the *same* base environment
    env_3_base = gym.make('InvertedDoublePendulum-v5')
    # Apply the wrapper to alter its task
    indices_to_mask = [3, 4]
    env_3 = PartialObservationWrapper(env_3_base, mask_indices=indices_to_mask)

    model_3 = PPO("MlpPolicy", env_3, verbose=0)

    callback_3 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='PenalizedDoublePendulum34'  # Updated label
    )

    print("Training Agent 3 (PenalizedDoublePendulum)...")
    model_3.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_2)
    print("Agent 3 training complete.")





    all_weights = np.array(callback_1.weights_log + callback_2.weights_log +callback_3.weights_log)
    all_weight_labels = np.array(callback_1.labels_log + callback_2.labels_log+callback_3.labels_log)
    all_weight_steps = np.array(callback_1.steps_log + callback_2.steps_log+callback_3.steps_log)

    # 2. NEW: Episode data
    all_ep_rewards = np.array(
        callback_1.ep_rewards_log + callback_2.ep_rewards_log +callback_3.ep_rewards_log)
    all_ep_lengths = np.array(
        callback_1.ep_lengths_log + callback_2.ep_lengths_log+callback_3.ep_lengths_log)
    all_ep_labels = np.array(
        callback_1.ep_labels_log + callback_2.ep_labels_log+callback_3.ep_labels_log)
    all_ep_steps = np.array(callback_1.ep_steps_log + callback_2.ep_steps_log+callback_3.ep_steps_log)

    # --- Save all arrays to a single compressed file ---

    output_filename = 'training_data.npz'  # Renamed file
    np.savez_compressed(
        output_filename,

        # Weight data
        weights=all_weights,
        weight_labels=all_weight_labels,
        weight_steps=all_weight_steps,

        # NEW: Episode data
        ep_rewards=all_ep_rewards,
        ep_lengths=all_ep_lengths,
        ep_labels=all_ep_labels,
        ep_steps=all_ep_steps
    )

    print(f"Successfully saved all training data to {output_filename}")


if __name__ == '__main__':
    main()
