import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import average_ordered_dicts, WeightStorageCallback,average_state_dicts,save_data
import random   
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS,clients_per_round,task_list, n_envs=16):


    # === Model Initialization ===

    # 1. Create the Global Model (Server Model)
    # It needs a dummy env just to initialize the network architecture
    dummy_env = gym.make("LunarLander-v3")
    global_model = PPO("MlpPolicy", dummy_env, verbose=0)
    dummy_env.close()

    # 2. Create Client Models, Envs, and Callbacks dynamically
    client_models = []
    client_envs = []
    client_callbacks = []

    print("Initializing clients...")
    for i, task in enumerate(task_list):
        label = task['label']
        gravity = task['gravity']
        wind = task['wind']
        print(f"  > Client {i+1} ({label}): Mask={gravity if gravity else 'None'}")

        # Create Env
        # env = gym.make('LunarLander-v3',gravity=gravity,enable_wind=True,wind_power=wind)
        env_kwargs = {'gravity': gravity, 'enable_wind': True, 'wind_power': wind}
        env = make_vec_env("LunarLander-v3", n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
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
    total_clients = len(client_models)
    print(f"\nStarting Federated Learning: {total_clients} clients, {NUM_ROUNDS} rounds, {LOCAL_STEPS} local steps per round.")

    # === Federated Training Loop ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        selected_indices = random.sample(range(total_clients), clients_per_round)
        selected_labels = [task_list[i]['label'] for i in selected_indices]
        print(f"Selected clients: {selected_labels}")
        active_client_state_dicts = []
        # 1. Broadcast global model weights to all clients
        global_state_dict = global_model.policy.state_dict()

        # 2. Local Training (loop over all clients)
        for idx in selected_indices:
            client = client_models[idx]
            callback = client_callbacks[idx]
            task_info = task_list[idx]

            # A. Load Global Weights (Sync)
            # In client selection, we only need to update the clients that are about to train
            client.policy.load_state_dict(global_state_dict)

            # B. Local Training
            print(f"  Training {task_info['label']}...")
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

            # C. Collect weights for aggregation
            active_client_state_dicts.append(client.policy.state_dict())
        
        avg_state_dict = average_state_dicts(active_client_state_dicts)

        # 4. Update global model
        global_model.policy.load_state_dict(avg_state_dict)

    print("\n--- Federated Learning Complete ---")
    #
    save_path = "models/fl_global_model_final.zip"
    global_model.save(save_path)
    print(f"Global model saved to {save_path}")

    # --- Data Collection and Saving (Dynamic) ---


    print("Collecting data from callbacks...")

    
    save_data(client_callbacks,'federated_training_data.npz')

    # Clean up all environments
    print("Closing environments...")
    for env in client_envs:
        env.close()
    print("All environments closed.")


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
            'label': 'Client_1_Moon',
            'gravity': -1.6,
            'wind': 0.5,
        },
        {
            'label': 'Client_2_Earth',
            'gravity': -9.8,
            'wind': 0.5,
        },
        {
            'label': 'Client_3_Mars',
            'gravity': -3.73,
            'wind': 0.5,
        },
        # You can add more clients just by editing this list!
        # {
        #     'label': 'Client_4_Standard_B',
        #     'mask': None
        # }
    ]

    # 3. Run the experiment
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, 2,fl_task_list, n_envs=16)

