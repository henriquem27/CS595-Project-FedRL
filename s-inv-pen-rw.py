import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO


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
    env_2 = CustomRewardWrapper(env_2_base, action_penalty=0.0)

    model_2 = PPO("MlpPolicy", env_2, verbose=0)

    callback_2 = WeightStorageCallback(
        check_freq=CHECK_FREQ,
        agent_label='PenalizedDoublePendulum'  # Updated label
    )

    print("Training Agent 2 (PenalizedDoublePendulum)...")
    model_2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_2)
    print("Agent 2 training complete.")

    # --- Data saving (no changes needed) ---


    all_weights = np.array(callback_1.weights_log + callback_2.weights_log)
    all_weight_labels = np.array(callback_1.labels_log + callback_2.labels_log)
    all_weight_steps = np.array(callback_1.steps_log + callback_2.steps_log)

    # 2. NEW: Episode data
    all_ep_rewards = np.array(
        callback_1.ep_rewards_log + callback_2.ep_rewards_log)
    all_ep_lengths = np.array(
        callback_1.ep_lengths_log + callback_2.ep_lengths_log)
    all_ep_labels = np.array(
        callback_1.ep_labels_log + callback_2.ep_labels_log)
    all_ep_steps = np.array(callback_1.ep_steps_log + callback_2.ep_steps_log)

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
