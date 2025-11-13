import torch as th
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from collections import OrderedDict
from helpers import average_ordered_dicts, WeightStorageCallback,average_state_dicts,save_data,average_deltas


def run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, DP_SENSITIVITY, DP_EPSILON):
    """
    Runs a federated learning experiment with differential privacy.

    Args:
        NUM_ROUNDS (int): Number of federation rounds.
        CHECK_FREQ (int): Frequency for callbacks to log data (in steps).
        LOCAL_STEPS (int): Number of training steps per client per round.
        task_list (list of dicts): Defines the clients.
        DP_SENSITIVITY (float): L1 sensitivity (clipping bound).
        DP_EPSILON (float): Privacy budget (epsilon).
    """

    # --- Differential Privacy Setup ---
    # The scale 'b' for Laplace noise is Sensitivity / Epsilon

    DP_SCALE = DP_SENSITIVITY / DP_EPSILON
    print(
        f"DP settings: Epsilon={DP_EPSILON}, Sensitivity={DP_SENSITIVITY}, Noise Scale={DP_SCALE:.4f}")

    # === Model Initialization ===

    # 1. Create the Global Model (Server Model)
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
            print(
                f"  > Client {i+1} ({label}): Mask={gravity if gravity else 'None'}")

            # Create Env
            env = gym.make('LunarLander-v3', gravity=gravity)
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

        # 1. Broadcast global model weights to clients
        # Keep a copy of the *original* global weights to calculate deltas
        global_state_dict = global_model.policy.state_dict()

        for client in client_models:
            client.policy.load_state_dict(global_state_dict)

        # 2. Local Training
        for i, (client, callback) in enumerate(zip(client_models, client_callbacks)):
            print(f"Training {task_list[i]['label']}...")
            client.learn(
                total_timesteps=LOCAL_STEPS,
                callback=callback,
                reset_num_timesteps=False
            )

        # -----------------------------------------------------------------

        print("Clipping and noising client deltas...")
        noisy_deltas = []

        for i, client in enumerate(client_models):
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
            print(
                f"    - {task_list[i]['label']} L1 norm: {total_l1_norm:.4f}")

            # --- Calculate the clipping factor ---
            # Clip factor = min(1, S / ||delta||_1)
            clip_factor = min(1.0, DP_SENSITIVITY / (total_l1_norm + 1e-6))

            # --- Second pass: Apply clipping and add Laplace noise ---
            for key in delta.keys():
                clipped_delta = delta[key] * clip_factor

                noise = th.tensor(
                    np.random.laplace(0, scale=DP_SCALE,
                                      size=clipped_delta.shape),
                    dtype=clipped_delta.dtype,
                    device=clipped_delta.device
                )
                noisy_delta[key] = clipped_delta + noise

            noisy_deltas.append(noisy_delta)

        # -----------------------------------------------------------------
        # 4. AGGREGATE and UPDATE
        # -----------------------------------------------------------------
        print("Aggregating noisy deltas...")
        avg_noisy_delta = average_ordered_dicts(noisy_deltas)

        # Update global model by *adding* the average noisy delta
        new_global_state_dict = OrderedDict()
        for key in global_state_dict.keys():
            new_global_state_dict[key] = global_state_dict[key] + \
                avg_noisy_delta[key]

        global_model.policy.load_state_dict(new_global_state_dict)

    print("\n--- Federated Learning Complete ---")
    save_data(client_callbacks, 'dp_training_data.npz')

    print("Closing environments...")
    for env in client_envs:
        env.close()


if __name__ == '__main__':

    print("Running dp_fl_training_utils.py as main script...")

    # === Define FL Hyperparameters ===
    NUM_ROUNDS = 20
    LOCAL_STEPS = 5000
    CHECK_FREQ = 5000  # Callback check frequency

    # === Define DP Hyperparameters ===
    DP_SENSITIVITY = 150.0  # L1 clipping norm (S)
    DP_EPSILON = 300.0      # Privacy budget

    # === Define the Clients (Tasks) ===
    # This list now drives the entire experiment
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
    ]
    # === Run the Experiment ===
    # (This was the missing piece in your original file)
    run_dp_fl_experiment(
        NUM_ROUNDS,
        CHECK_FREQ,
        LOCAL_STEPS,
        fl_task_list,
        DP_SENSITIVITY,
        DP_EPSILON
    )
