import gymnasium as gym  # Use Gymnasium, as it's the successor to Gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from helpers import WeightStorageCallback, save_data


def run_experiment_training(TOTAL_TIMESTEPS, CHECK_FREQ, task_list):

    client_callbacks = []

    # --- Loop over all defined tasks ---
    for i, task in enumerate(task_list):
        agent_num = i + 1
        gravity = task['gravity']
        agent_label = task['label']

        print(f"--- Training Agent {agent_num} ({agent_label}) ---")

        # 1. Create the environment
        env = gym.make("LunarLander-v3", gravity=gravity)


        # 2. Create the model and callback
        model = PPO("MlpPolicy", env, verbose=0)

        callback = WeightStorageCallback(
            check_freq=CHECK_FREQ,
            agent_label=agent_label
        )
        client_callbacks.append(callback)
        # 3. Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        print(f"Agent {agent_num} ({agent_label}) training complete.")

        env.close()  # Good practice to close envs

    # --- Aggregate all data into numpy arrays ---
    

    # --- Save all arrays to a single compressed file ---
    output_filename = 'training_data.npz'
    save_data(
        client_callbacks,output_filename
    )


if __name__ == '__main__':

    task_list = [
        {
            'label': 'Client_1_Moon',
            'gravity': -1.6,
        },
        {
            'label': 'Client_2_Earth',
            'gravity': -9.8,
        },
        {
            'label': 'Client_3_Mars',
            'gravity': -3.73,
        },
        # You can add more clients just by editing this list!
        # {
        #     'label': 'Client_4_Standard_B',
        #     'mask': None
        # }
    ]
    run_experiment_training(TOTAL_TIMESTEPS=50000,
                            CHECK_FREQ=500, task_list=task_list)
