import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch as th
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
import random
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import ExperimentLogger, StreamingCallback
# Import your existing helpers
from helpers import average_state_dicts, WeightStorageCallback, save_data

# --- 1. THE DYNAMIC WRAPPER ---
class DynamicLunarLander(LunarLander):
    """
    Wraps LunarLander to allow changing physics (Gravity/Wind)
    without destroying the process.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reconfigure(self, gravity, wind_power):
        """
        Updates physics parameters.
        The next time reset() is called, these take effect.
        """
        self.gravity = gravity
        self.enable_wind = True
        self.wind_power = wind_power
        return True

def run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, clients_per_round, task_list, experiment_name="fl_run"):
    
    # --- 2. SETUP RESOURCES ---
    MAX_CPUS = os.cpu_count()
    N_ENVS = 32 # Leave 2 cores for overhead
    
    print(f"System detected {MAX_CPUS} CPUs. Using {N_ENVS} Persistent Environments.")

    # --- 3. INITIALIZE LOGGER & GLOBAL MODEL ---
    # Initialize the disk logger
    logger = ExperimentLogger(experiment_name=experiment_name)
    
    dummy_env = make_vec_env(DynamicLunarLander, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    global_model = PPO("MlpPolicy", dummy_env, verbose=0)
    
    # --- 4. START THE WORKER POOL ---
    print(f"Spinning up {N_ENVS} persistent worker processes...")
    training_vec_env = make_vec_env(
        DynamicLunarLander,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'gravity': -10.0, 'enable_wind': True, 'wind_power': 0.0}
    )

    # --- 5. INITIALIZE CLIENT PLACEHOLDERS ---
    client_models = []
    # We no longer need a list of callbacks, we create them on the fly per round
    
    print("Initializing client structures...")
    for i, task in enumerate(task_list):
        client = PPO("MlpPolicy", dummy_env, verbose=0)
        client_models.append(client)
    
    total_clients = len(client_models)
    print(f"\nStarting Federated Learning: {total_clients} clients, {NUM_ROUNDS} rounds. Logging to /logs/{experiment_name}")

    # === FEDERATED TRAINING LOOP ===
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")

        selected_indices = random.sample(range(total_clients), clients_per_round)
        selected_labels = [task_list[i]['label'] for i in selected_indices]
        print(f"Selected clients: {selected_labels}")

        active_client_state_dicts = []
        
        # Broadcast global weights
        global_state_dict = global_model.policy.state_dict()

        for idx in selected_indices:
            client = client_models[idx]
            task_info = task_list[idx]

            # A. Sync Weights
            client.policy.load_state_dict(global_state_dict)
            print("Synced weights for client:", task_info['label'])
            # B. HOT-SWAP ENVIRONMENT
            training_vec_env.env_method(
                "reconfigure", 
                gravity=task_info['gravity'], 
                wind_power=task_info['wind']
            )
            training_vec_env.reset()
            client.set_env(training_vec_env)

            # C. CREATE STREAMING CALLBACK
            # This callback will write rewards to CSV immediately during training
            callback = StreamingCallback(
                logger=logger,
                agent_label=task_info['label'],
                round_num=round_num
            )

            # D. Local Training
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

            # E. SAVE WEIGHTS TO DISK & COLLECT FOR AGGREGATION
            current_weights = client.policy.state_dict()
            
            # Save to: logs/exp_name/weights/round_X/Client_Y.pt
            logger.save_client_weights(task_info['label'], round_num, current_weights)
            
            active_client_state_dicts.append(current_weights)

        # 3. Aggregate (Standard FedAvg)
        if active_client_state_dicts:
            print("  > Aggregating weights...")
            avg_state_dict = average_state_dicts(active_client_state_dicts)
            global_model.policy.load_state_dict(avg_state_dict)
            
            # Optional: Save global model for this round too
            logger.save_client_weights("Global_Model", round_num, avg_state_dict)

    print("\n--- Federated Learning Complete ---")
    
    # Save Final Global Model
    final_save_path = f"logs/{experiment_name}/global_model_final.zip"
    global_model.save(final_save_path)
    print(f"Final global model saved to {final_save_path}")

    # Cleanup
    training_vec_env.close()
    dummy_env.close()

if __name__ == '__main__':
    print("Running Optimized Baseline FL...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 50
    LOCAL_STEPS = 25000  
    CHECK_FREQ = 2000   
    CLIENTS_PER_ROUND = 16 # Safe to use higher numbers now

    # 2. Define the clients
    # (Example List - Ensure you use your full list from the previous file)
    fl_task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.62, 'wind': 0.5},
        {'label': 'Client_2_Earth', 'gravity': -9.81, 'wind': 2.0},
        {'label': 'Client_3_Mars', 'gravity': -3.71, 'wind': 0.1},
        # ... Add all your other clients here ...
    ]

    # 3. Run
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, CLIENTS_PER_ROUND, fl_task_list)