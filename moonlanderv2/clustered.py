import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import (
    average_ordered_dicts,
    WeightStorageCallback,
    save_data,
    cluster_and_average_models,  # <-- Import new function
    find_closest_model           # <-- Import new function
)


def run_fl_experiment_clustered(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, N_CLUSTERS):
    """
    Runs a Clustered Federated Learning experiment.
    
    N_CLUSTERS: Number of "global" models to maintain (e.g., 3 for Moon/Earth/Mars)
    """

    # === Model Initialization ===
    print(f"Initializing {N_CLUSTERS} global models for clustering...")
    dummy_env = gym.make("LunarLander-v3")
    # We now have a LIST of global models, one per cluster
    global_models = [PPO("MlpPolicy", dummy_env, verbose=0)
                     for _ in range(N_CLUSTERS)]
    dummy_env.close()

    # 2. Create Client Models, Envs, and Callbacks (same as before)
    client_models = []
    client_envs = []
    client_callbacks = []

    print("Initializing clients...")
    for i, task in enumerate(task_list):
        label = task['label']
        gravity = task['gravity']
        wind = task['wind']
        print(f"  > Client {i+1} ({label})")
        env = gym.make('LunarLander-v3', gravity=gravity,
                       enable_wind=True, wind_power=wind)
        client_envs.append(env)
        client = PPO("MlpPolicy", env, verbose=0)
        client_models.append(client)
        callback = WeightStorageCallback(
            check_freq=CHECK_FREQ, agent_label=label)
        client_callbacks.append(callback)

    print(
        f"\nStarting Clustered FL: {len(task_list)} clients, {N_CLUSTERS} clusters, {NUM_ROUNDS} rounds.")

    # === Federated Training Loop ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        # 1. BROADCAST & ASSIGNMENT (This is the first major change)
        print("Broadcasting models and assigning clients to clusters...")
        global_state_dicts = [model.policy.state_dict()
                              for model in global_models]

        assignments = []
        for i, client in enumerate(client_models):
            # Client compares its current model to all global models
            client_state_dict = client.policy.state_dict()

            # Find the global model "closest" to this client
            best_model_idx = find_closest_model(
                client_state_dict, global_state_dicts)
            assignments.append(best_model_idx)

            # Client pulls the weights from its assigned cluster's model
            client.policy.load_state_dict(global_state_dicts[best_model_idx])

        # Optional: Print cluster assignment summary
        for c_idx in range(N_CLUSTERS):
            count = np.sum(np.array(assignments) == c_idx)
            print(f"  Global Model {c_idx} assigned to {count} clients.")

        # 2. LOCAL TRAINING (Same as before)
        for i, (client, callback) in enumerate(zip(client_models, client_callbacks)):
            print(
                f"Training {task_list[i]['label']} (assigned to cluster {assignments[i]})...")
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

        # 3. AGGREGATION (This is the second major change)
        print("Aggregating models via clustering...")
        client_state_dicts = [client.policy.state_dict()
                              for client in client_models]

        # Use the new clustering function
        new_global_state_dicts = cluster_and_average_models(
            client_state_dicts,
            n_clusters=N_CLUSTERS
        )

        # 4. UPDATE GLOBAL MODELS
        # If clustering worked, update models.
        if len(new_global_state_dicts) == N_CLUSTERS:
            for i, new_dict in enumerate(new_global_state_dicts):
                global_models[i].policy.load_state_dict(new_dict)
        else:
            # Fallback (e.g., if clustering returned < N_CLUSTERS models)
            print("Clustering fallback: Averaging all clients into all global models.")
            avg_all = average_ordered_dicts(client_state_dicts)
            for model in global_models:
                model.policy.load_state_dict(avg_all)

    print("\n--- Clustered Federated Learning Complete ---")

    save_data(client_callbacks, 'clustered_fl_training_data.npz')

    print("Closing environments...")
    for env in client_envs:
        env.close()


if __name__ == '__main__':

    print("Running fl_moon_clustered.py as main script...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 5
    LOCAL_STEPS = 10000
    CHECK_FREQ = 2000
    N_CLUSTERS = 3  # <-- We know we have "3" environments

    # 2. Define the clients (Your full list from experiment.py)
    fl_task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.6,'wind': 0.5},
        {'label': 'Client_2_Earth', 'gravity': -9.8,'wind': 0.5},
        {'label': 'Client_3_Mars', 'gravity': -3.73,'wind': 0.5},
        # ... add all 33 clients ...
    ]
    # 3. Run the experiment
    run_fl_experiment_clustered(
        NUM_ROUNDS,
        CHECK_FREQ,
        LOCAL_STEPS,
        fl_task_list,
        N_CLUSTERS
    )
