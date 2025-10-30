import gymnasium as gym
from gymnasium.spaces import Box
import gymnasium as gym # Use Gymnasium, as it's the successor to Gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import gymnasium_robotics
class WeightStorageCallback(BaseCallback):
    """
    A custom callback to store model weights during training.

    :param check_freq: How often (in terms of environment steps) to store the weights.
    :param agent_label: A label for the agent type (e.g., 'InvertedPendulum' or 'DoubleInvertedPendulum').
    """

    def __init__(self, check_freq: int, agent_label: str, verbose: int = 0):
        super(WeightStorageCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.agent_label = agent_label

        # Lists to store the collected data
        self.weights_log = []
        self.labels_log = []
        self.steps_log = []

    def _on_step(self) -> bool:
        """
        This method is called after each environment step.
        """
        # Check if it's time to store the weights
        if self.n_calls % self.check_freq == 0:

            # Get the model's state dictionary (all weights and biases)
            state_dict = self.model.policy.state_dict()

            # Flatten all parameters into a single 1D numpy vector.
            # We move tensors to CPU (.cpu()) before converting to numpy (.numpy()).
            flat_weights = np.concatenate([
                param.cpu().detach().numpy().flatten()
                for param in state_dict.values()
            ])

            # Store the data
            self.weights_log.append(flat_weights)
            self.labels_log.append(self.agent_label)
            self.steps_log.append(self.n_calls)

            if self.verbose > 0:
                print(
                    f"Step {self.n_calls}: Stored weights for {self.agent_label} (vector size: {flat_weights.shape[0]})")

        # Return True to continue training
        return True


class PadObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to pad the observation space to a desired shape.
    
    :param env: The environment to wrap
    :param desired_shape: The target shape (e.g., (11,))
    """

    def __init__(self, env, desired_shape):
        super(PadObservationWrapper, self).__init__(env)
        self.desired_shape = desired_shape

        # Sanity check: ensure the original shape is smaller
        original_shape_flat = np.prod(self.observation_space.shape)
        desired_shape_flat = np.prod(self.desired_shape)
        assert original_shape_flat < desired_shape_flat, "Original obs space must be smaller than desired shape"

        # Define the new observation space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.desired_shape,
            dtype=self.observation_space.dtype
        )

    def _pad_obs(self, obs):
        """Pads a single observation."""
        # Flatten the original observation
        obs_flat = obs.flatten()
        # Create a new array of zeros with the desired shape
        padded_obs = np.zeros(self.desired_shape, dtype=obs.dtype)
        # Copy the original observation data into the start of the new array
        padded_obs[:len(obs_flat)] = obs_flat
        return padded_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pad_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

class CustomRewardWrapper(gym.Wrapper):
    """
    This wrapper penalizes large actions to create a "slightly altered task".
    
    :param env: The environment to wrap
    :param action_penalty: The coefficient for the action penalty
    """

    def __init__(self, env, action_penalty=0.1):
        super(CustomRewardWrapper, self).__init__(env)
        self.action_penalty = action_penalty

    def step(self, action):
        # Call the original step method
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add our custom penalty
        # We penalize the squared magnitude of the action
        penalty = -self.action_penalty * np.sum(np.square(action))
        new_reward = reward + penalty

        return obs, new_reward, terminated, truncated, info


# (Your WeightStorageCallback class definition goes here)


def main():
    TOTAL_TIMESTEPS = 100000
    CHECK_FREQ = 5000

    # === Define the Environments ===

    # Agent 2 (The "larger" one) - no changes
    env_2_base = gym.make('InvertedDoublePendulum-v2')
    # Get its observation shape, which we will use as the target
    target_obs_shape = env_2_base.observation_space.shape
    print(
        f"Target observation shape (from DoublePendulum): {target_obs_shape}")

    # Agent 1 (The "smaller" one) - We wrap it
    env_1_base = gym.make('InvertedPendulum-v4')
    print(
        f"Original observation shape (from Pendulum): {env_1_base.observation_space.shape}")

    # Apply the wrapper to pad Agent 1's observations
    env_1 = PadObservationWrapper(env_1_base, target_obs_shape)
    print(
        f"NEW observation shape (from Pendulum): {env_1.observation_space.shape}")

    # We also use the unwrapped env_2_base for Agent 2
    env_2 = env_2_base

    # === Create and Train Models ===
    # Because env_1 and env_2 now report identical observation and action spaces,
    # SB3 will create identical "MlpPolicy" models for them.

    # Agent 1 (Padded Inverted Pendulum)
    model_1 = PPO("MlpPolicy", env_1, verbose=0)
    callback_1 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='InvertedPendulum'
    )

    print("\nTraining Agent 1 (InvertedPendulum)...")
    model_1.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_1)
    print("Agent 1 training complete.")

    # Agent 2 (Double Inverted Pendulum)
    model_2 = PPO("MlpPolicy", env_2, verbose=0)
    callback_2 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='DoubleInvertedPendulum'
    )

    print("\nTraining Agent 2 (DoubleInvertedPendulum)...")
    model_2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_2)
    print("Agent 2 training complete.")

    # --- This part will now work! ---

    print("\nTraining complete. Combining and saving data...")
    all_weights = np.array(callback_1.weights_log + callback_2.weights_log)
    all_labels = np.array(callback_1.labels_log + callback_2.labels_log)
    all_steps = np.array(callback_1.steps_log + callback_2.steps_log)

    print(f"Successfully combined data.")
    # This will now be a valid 2D shape
    print(f"Data shape (samples, features): {all_weights.shape}")

    # Save to file
    output_filename = 'training_weights.npz'
    np.savez_compressed(
        output_filename,
        weights=all_weights,
        labels=all_labels,
        steps=all_steps
    )

    print(f"Successfully saved data to {output_filename}")


if __name__ == '__main__':
    main()
