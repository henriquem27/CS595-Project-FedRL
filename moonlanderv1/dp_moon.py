import torch as th
import gymnasium as gym
# import gymnasium_robotics # Uncomment if needed, but not used in snippet
import numpy as np
import random # <--- NEW IMPORT
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import average_ordered_dicts, WeightStorageCallback, average_state_dicts, save_data, average_deltas


def run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, DP_SENSITIVITY, DP_EPSILON, clients_per_round):
    """
    Runs a federated learning experiment with differential privacy and client selection.
    """

    # --- Differential Privacy Setup ---
    DP_SCALE = DP_SENSITIVITY / DP_EPSILON
    print(f"DP settings: Epsilon={DP_EPSILON}, Sensitivity={DP_SENSITIVITY}, Noise Scale={DP_SCALE:.4f}")

    # === Model Initialization ===

    # 1. Create the Global Model (Server Model)
    dummy_env = gym.make("LunarLander-v3")
    global_model = PPO("MlpPolicy", dummy_env, verbose=0)
    dummy_env.close()

    # 2. Create Client Models, Envs, and Callbacks dynamically
    client_models = []
    client_envs = []
    client_callbacks = []
    
    # Removed the conflicting list initialization 'clients_per_round = []' from original snippet
    
    print("Initializing clients...")

    for i, task in enumerate(task_list):
        label = task['label']
        gravity = task['gravity']
        wind = task['wind']
        print(f"  > Client {i+1} ({label}): Mask={gravity if gravity else 'None'}")

        # Create Env
        env = gym.make('LunarLander-v3', gravity=gravity, enable_wind=True, wind_power=wind)
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

    total_clients = len(task_list)
    print(f"\nStarting Federated Learning: {total_clients} clients, selecting {clients_per_round} per round.")

    # === Federated Training Loop ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        # 1. Select Clients
        selected_indices = random.sample(range(total_clients), clients_per_round)
        print(f"Selected client indices: {selected_indices}")

        # Keep a copy of the *original* global weights to calculate deltas
        global_state_dict = global_model.policy.state_dict()
        
        # List to store noisy deltas ONLY from selected clients
        active_noisy_deltas = []

        # 2. Loop ONLY through selected clients
        for idx in selected_indices:
            client = client_models[idx]
            callback = client_callbacks[idx]
            task_label = task_list[idx]['label']

            # A. Sync: Load global weights to this client
            client.policy.load_state_dict(global_state_dict)

            # B. Local Training
            print(f"  Training {task_label}...")
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

            # C. DP Logic: Calculate Delta -> Clip -> Noise
            # Get the client's new weights
            new_weights = client.policy.state_dict()

            delta = OrderedDict()
            noisy_delta = OrderedDict()
            total_l1_norm = 0.0

            # --- First pass: Calculate delta and its total L1 norm ---
            for key in global_state_dict.keys():
                delta[key] = new_weights[key] - global_state_dict[key]
                total_l1_norm += th.sum(th.abs(delta[key]))

            total_l1_norm = total_l1_norm.item()
            
            # Clip factor = min(1, S / ||delta||_1)
            clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))

            # --- Second pass: Apply clipping and add Laplace noise ---
            for key in delta.keys():
                clipped_delta = delta[key] * clip_factor

                noise = th.tensor(
                    np.random.laplace(0, scale=DP_SCALE, size=clipped_delta.shape),
                    dtype=clipped_delta.dtype,
                    device=clipped_delta.device
                )
                noisy_delta[key] = clipped_delta + noise

            active_noisy_deltas.append(noisy_delta)

        # -----------------------------------------------------------------
        # 3. AGGREGATE and UPDATE
        # -----------------------------------------------------------------
        print(f"Aggregating noisy deltas from {len(active_noisy_deltas)} clients...")
        
        # We average the deltas from the *selected* clients
        avg_noisy_delta = average_ordered_dicts(active_noisy_deltas)

        # Update global model by *adding* the average noisy delta
        new_global_state_dict = OrderedDict()
        for key in global_state_dict.keys():
            new_global_state_dict[key] = global_state_dict[key] + avg_noisy_delta[key]

        global_model.policy.load_state_dict(new_global_state_dict)

    # === SAVE THE GLOBAL MODEL ===
    # This is the fix you requested to save the model for GIFs
    model_save_name = f"models/dp_global_model_ep{int(DP_EPSILON)}.zip"
    global_model.save(model_save_name)
    print(f"\nGlobal model saved to: {model_save_name}")

    print("--- Federated Learning Complete ---")
    
    # Save data
    filename = f"dp_training_data_ep{int(DP_EPSILON)}_sens{int(DP_SENSITIVITY)}.npz"
    save_data(client_callbacks, filename)

    print("Closing environments...")
    for env in client_envs:
        env.close()


if __name__ == '__main__':

    print("Running dp_fl_training_utils.py as main script...")

    # === Define FL Hyperparameters ===
    NUM_ROUNDS = 3
    LOCAL_STEPS = 5000
    CHECK_FREQ = 5000
    
    # NEW: Number of clients to select per round
    CLIENTS_PER_ROUND = 2 

    # === Define DP Hyperparameters ===
    DP_SENSITIVITY = 15.0
    DP_EPSILON = 30.0

    # === Define the Clients (Tasks) ===
    fl_task_list = [
        {
            'label': 'Client_1_Moon',
            'gravity': -1.6,
            'wind': 0.0,
        },
        {
            'label': 'Client_2_Earth',
            'gravity': -9.8,
            'wind': 0.5,
        },
        {
            'label': 'Client_3_Mars',
            'gravity': -3.73,
            'wind': 0.2,
        },
    ]
    
    # Verify we aren't asking for more clients than exist
    if CLIENTS_PER_ROUND > len(fl_task_list):
        print("Error: CLIENTS_PER_ROUND cannot be larger than the total number of clients.")
    else:
        # === Run the Experiment ===
        run_dp_fl_experiment(
            NUM_ROUNDS,
            CHECK_FREQ,
            LOCAL_STEPS,
            fl_task_list,
            DP_SENSITIVITY,
            DP_EPSILON,
            CLIENTS_PER_ROUND # Passed correctly here
        )