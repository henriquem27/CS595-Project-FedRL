import gymnasium as gym
import numpy as np
import torch as th
import os
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Ensure you have updated helpers.py with the ExperimentLogger class
from helpers import ExperimentLogger, StreamingCallback

# --- Custom Callback for Single Agent Checkpointing ---
class PeriodicSaveCallback(BaseCallback):
    """
    Saves the model weights to disk every `check_freq` steps.
    """
    def __init__(self, logger, agent_label, check_freq, verbose=0):
        super(PeriodicSaveCallback, self).__init__(verbose)
        # FIXED: Renamed self.logger -> self.exp_logger to avoid SB3 conflict
        self.exp_logger = logger
        self.agent_label = agent_label
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save weights. We use 'n_calls' (steps) as the round number
            # so folders will look like: logs/.../weights/round_5000/Client_1.pt
            self.exp_logger.save_client_weights(
                client_label=self.agent_label,
                round_num=self.n_calls,
                state_dict=self.model.policy.state_dict()
            )
            if self.verbose > 0:
                print(f"Saved checkpoint for {self.agent_label} at step {self.n_calls}")
        return True

def run_experiment_training(TOTAL_TIMESTEPS, CHECK_FREQ, task_list, experiment_name="single_agent_run"):
    
    # --- 1. Initialize Logger ---
    # This creates logs/single_agent_run/metrics and logs/single_agent_run/weights
    logger = ExperimentLogger(experiment_name=experiment_name)
    N_ENVS = 32
    print(f"Starting Single Agent Experiment. Logging to /logs/{experiment_name}")

    # --- Loop over all defined tasks ---
    for i, task in enumerate(task_list):
        agent_num = i + 1
        gravity = task.get('gravity', -10.0) # Default if missing
        wind = task.get('wind', 0.0)         # Default if missing
        agent_label = task['label']

        print(f"\n--- Training Agent {agent_num} ({agent_label}) ---")

        # 2. Create the environment
        try:
            # Try passing kwargs (requires LunarLander-v3 or compatible wrapper)
            env = make_vec_env(
                "LunarLander-v3",
                n_envs=N_ENVS,
                env_kwargs={'gravity': gravity, 'enable_wind': True, 'wind_power': wind}
            )
        except:
            print("Standard LunarLander-v3 didn't accept kwargs, using default env for safety.")
            env = gym.make('LunarLander-v3')

        # 3. Create the model
        model = PPO("MlpPolicy", env, verbose=0)

        # 4. Create Callbacks
        # A. Streaming: Logs rewards to CSV immediately
        metrics_callback = StreamingCallback(
            logger=logger, 
            agent_label=agent_label, 
            round_num=0 
        )
        
        # B. Periodic Saver: Saves weights to disk every X steps
        saver_callback = PeriodicSaveCallback(
            logger=logger,
            agent_label=agent_label,
            check_freq=CHECK_FREQ
        )
        
        # Combine them
        callback_list = CallbackList([metrics_callback, saver_callback])

        # 5. Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
        
        # 6. Save Final Model
        logger.save_client_weights(
            client_label=agent_label, 
            round_num=TOTAL_TIMESTEPS, 
            state_dict=model.policy.state_dict()
        )
        
        print(f"Agent {agent_num} ({agent_label}) training complete.")
        env.close() 

    print(f"\nAll agents finished. Data saved to logs/{experiment_name}")

if __name__ == '__main__':

    task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.6, 'wind': 0.0},
        {'label': 'Client_2_Earth', 'gravity': -9.8, 'wind': 0.0},
        {'label': 'Client_3_Mars', 'gravity': -3.73, 'wind': 0.0},
    ]
    
    run_experiment_training(
        TOTAL_TIMESTEPS=50000,
        CHECK_FREQ=2000, 
        task_list=task_list,
        experiment_name="single_agent_baseline"
    )