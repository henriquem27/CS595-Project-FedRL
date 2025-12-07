import torch as th
import gymnasium as gym
import numpy as np
import os
import random
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from collections import OrderedDict

# Import the new logging classes
from helpers import average_ordered_dicts, ExperimentLogger, StreamingCallback
from helpers import DynamicLunarLander

# --- THREADING SETTINGS ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, DP_SENSITIVITY, DP_EPSILON, clients_per_round, experiment_name="dp_fl_run"):
    
    experiment_name = f"dp_fl_sens{DP_SENSITIVITY}_eps{DP_EPSILON}"

    # --- 1. SETUP ---
    MAX_CPUS = os.cpu_count()
    N_ENVS = 64  # Safety buffer
    
    print(f"System detected {MAX_CPUS} CPUs. Using {N_ENVS} Persistent Environments.")

    # --- 2. INITIALIZE LOGGER & GLOBAL MODEL ---
    # Create the logger (folders: logs/exp_name/weights/ & logs/exp_name/metrics/)
    logger = ExperimentLogger(experiment_name=experiment_name)

    dummy_env = make_vec_env(DynamicLunarLander, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    global_model = PPO("MlpPolicy", dummy_env, verbose=0)
    
    # --- 3. INITIALIZE PERSISTENT TRAINING POOL ---
    print(f"Spinning up {N_ENVS} persistent worker processes...")
    training_vec_env = make_vec_env(
        DynamicLunarLander,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'gravity': -10.0, 'enable_wind': True, 'wind_power': 0.0}
    )

    # Initialize Client Models (Placeholders)
    client_models = []
    # We no longer need 'client_callbacks' list; we create them dynamically per round.
    print("Initializing client placeholders...")
    for i, task in enumerate(task_list):
        client = PPO("MlpPolicy", dummy_env, verbose=0)
        client_models.append(client)

    total_clients = len(task_list)
    print(f"\nStarting DP-FL: {total_clients} clients, {NUM_ROUNDS} rounds. Logging to /logs/{experiment_name}")

    # === FEDERATED LOOP ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        selected_indices = random.sample(range(total_clients), clients_per_round)
        selected_labels = [task_list[i]['label'] for i in selected_indices]
        print(f"Selected: {selected_labels}")

        global_state_dict = global_model.policy.state_dict()
        active_noisy_deltas = []

        for idx in selected_indices:
            client = client_models[idx]
            task_info = task_list[idx]

            # A. Sync Weights
            client.policy.load_state_dict(global_state_dict)

            # B. HOT-SWAP THE ENVIRONMENT
            training_vec_env.env_method(
                "reconfigure", 
                gravity=task_info['gravity'], 
                wind_power=task_info['wind']
            )
            training_vec_env.reset()
            client.set_env(training_vec_env)

            # C. SETUP STREAMING LOGGING
            # Create a callback specific to this client/round
            callback = StreamingCallback(
                logger=logger, 
                agent_label=task_info['label'], 
                round_num=round_num
            )

            # D. Train
            client.learn(total_timesteps=LOCAL_STEPS, callback=callback, reset_num_timesteps=False)

            # E. SAVE WEIGHTS (Pre-DP)
            # Save the actual trained weights to disk before we noise them
            current_weights = client.policy.state_dict()
            logger.save_client_weights(task_info['label'], round_num, current_weights)

            # F. DP Logic (Compute Delta -> Clip -> Add Noise)
            delta = OrderedDict()
            noisy_delta = OrderedDict()
            total_l1_norm = 0.0

            # Calculate update (Delta)
            for key in global_state_dict.keys():
                delta[key] = current_weights[key] - global_state_dict[key]
                total_l1_norm += th.sum(th.abs(delta[key]))

            total_l1_norm = total_l1_norm.item()
            clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))
            DP_SCALE = DP_SENSITIVITY / DP_EPSILON

            for key in delta.keys():
                clipped_delta = delta[key] * clip_factor
                noise = th.tensor(np.random.laplace(0, scale=DP_SCALE, size=clipped_delta.shape),
                                  dtype=clipped_delta.dtype, device=clipped_delta.device)
                noisy_delta[key] = clipped_delta + noise

            active_noisy_deltas.append(noisy_delta)

        # Aggregation
        if active_noisy_deltas:
            print(f"  Aggregating {len(active_noisy_deltas)} updates...")
            avg_noisy_delta = average_ordered_dicts(active_noisy_deltas)
            
            # Apply averaged noisy delta to global model
            new_global_state_dict = OrderedDict()
            for key in global_state_dict.keys():
                new_global_state_dict[key] = global_state_dict[key] + avg_noisy_delta[key]
            
            global_model.policy.load_state_dict(new_global_state_dict)

            # Optional: Save global model for this round
            logger.save_client_weights("Global_Model", round_num, new_global_state_dict)

    # === CLEANUP ===
    print("Closing persistent environments...")
    training_vec_env.close()
    dummy_env.close()
    
    # Save Final Model
    final_path = f"logs/{experiment_name}/global_model_final.zip"
    global_model.save(final_path)
    print(f"Final model saved to {final_path}")

if __name__ == '__main__':

    print("Running dp_fl_experiment with Disk Logging...")

    # === Define FL Hyperparameters ===
    NUM_ROUNDS = 3
    LOCAL_STEPS = 5000
    CHECK_FREQ = 5000 # Note: Used by logger internally if needed, but StreamingCallback logs every episode end
    CLIENTS_PER_ROUND = 2 

    # === Define DP Hyperparameters ===
    DP_SENSITIVITY = 15.0
    DP_EPSILON = 30.0

    # === Define the Clients (Tasks) ===
    fl_task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.6, 'wind': 0.0},
        {'label': 'Client_2_Earth', 'gravity': -9.8, 'wind': 0.5},
        {'label': 'Client_3_Mars', 'gravity': -3.73, 'wind': 0.2},
    ]
    
    run_dp_fl_experiment(
        NUM_ROUNDS=NUM_ROUNDS,
        CHECK_FREQ=CHECK_FREQ,
        LOCAL_STEPS=LOCAL_STEPS,
        task_list=fl_task_list,
        DP_SENSITIVITY=DP_SENSITIVITY,
        DP_EPSILON=DP_EPSILON,
        clients_per_round=CLIENTS_PER_ROUND,
        experiment_name=f"dp_fl_sens{DP_SENSITIVITY}_eps{DP_EPSILON}"
    )