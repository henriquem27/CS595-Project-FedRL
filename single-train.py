import gymnasium as gym # Use Gymnasium, as it's the successor to Gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
"""
This script trains two PPO agents on the InvertedPendulum-v5 environment.
One agent is trained on the standard task, while the other is trained on a slightly altered task
by modifying the reward function to penalize large actions.
The script collects the model weights at regular intervals during training and saves them
to a .npz file for later analysis.
"""
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


def main():
    TOTAL_TIMESTEPS = 100000
    CHECK_FREQ = 5000

    # === Agent 1 (Standard Task) ===
    # Both agents now use the SAME environment
    env_1 = gym.make('InvertedPendulum-v5')
    model_1 = PPO("MlpPolicy", env_1, verbose=0)

    callback_1 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='StandardPendulum'  # Renamed label
    )

    print("Training Agent 1 (StandardPendulum)...")
    model_1.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_1)
    print("Agent 1 training complete.")

    # === Agent 2 (Altered Task) ===
    # We use the same base env, but wrap it to change the reward
    env_2_base = gym.make('InvertedPendulum-v5')
    env_2 = CustomRewardWrapper(
        env_2_base, action_penalty=0.1)  # Apply wrapper

    model_2 = PPO("MlpPolicy", env_2, verbose=0)

    callback_2 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='PenalizedPendulum'  # Renamed label
    )

    print("Training Agent 2 (PenalizedPendulum)...")
    model_2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_2)
    print("Agent 2 training complete.")

    # --- This part will now work! ---

    print("\nTraining complete. Combining and saving data...")

    # Combine data
    all_weights = np.array(callback_1.weights_log + callback_2.weights_log)
    all_labels = np.array(callback_1.labels_log + callback_2.labels_log)
    all_steps = np.array(callback_1.steps_log + callback_2.steps_log)

    print(f"Successfully combined data.")
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
