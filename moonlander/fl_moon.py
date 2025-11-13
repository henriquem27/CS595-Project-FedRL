import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import average_ordered_dicts, WeightStorageCallback,average_state_dicts,save_data


def run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list):


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
        print(f"  > Client {i+1} ({label}): Mask={gravity if gravity else 'None'}")

        # Create Env
        env = gym.make('LunarLander-v3',gravity=gravity)
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

    print(f"\nStarting Federated Learning: {len(task_list)} clients, {NUM_ROUNDS} rounds, {LOCAL_STEPS} local steps per round.")

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

    # 3. Run the experiment
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, fl_task_list)
