
# --- Imports ---
from fl_moon import run_fl_experiment
from single_moon import run_experiment_training
from dp_moon import run_dp_fl_experiment  
import pprint
# from clustered import run_fl_experiment_clustered # Uncomment if needed
import random
import time

import multiprocessing
import torch # Assuming you use PyTorch
import gc

def run_isolated(target_func, kwargs):
    """
    Runs a function in a separate process to ensure complete resource 
    cleanup (GPU, RAM, Threads) upon completion.
    """
    # Create the process
    p = multiprocessing.Process(target=target_func, kwargs=kwargs)
    p.start()
    p.join() # Wait for the process to finish
    
    # Check exit code. 0 means success, anything else is an error.
    if p.exitcode != 0:
        raise RuntimeError(f"Process for {target_func.__name__} failed with exit code {p.exitcode}")
    
    # Optional: Cleanup in the main process just in case
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def add_derived_tasks(original_list, num_to_add_per_task):
    new_items = []
    # Iterate over each task in the *original* list
    for task in original_list:
        new_task_base = task.copy()

        # Try to parse the label
        try:
            parts = new_task_base['label'].split('_')
            base_name = parts[0]  # e.g., "Client"
            current_id = int(parts[1])  # e.g., 1
            suffix = parts[2]  # e.g., "Moon"
        except (IndexError, ValueError, TypeError):
            print(f"Skipping task with unexpected label: {task.get('label')}")
            continue

        # Create the specified number of new tasks
        for i in range(1, num_to_add_per_task + 1):
            new_task = new_task_base.copy()
            new_id = current_id - i
            random_number = random.randint(0, 15)
            new_task['label'] = f"{base_name}_{new_id}_{suffix}"
            new_task['wind'] = random_number 
            new_items.append(new_task)

    original_list.extend(new_items)
    print(f"Added {len(new_items)} new tasks.")

if __name__ == "__main__":
    print("Running experiment.py sequentially (No Parallelism)...")
    execution_times = []

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 2
    LOCAL_STEPS = 50000
    CHECK_FREQ = 2000    
    total_steps = NUM_ROUNDS * LOCAL_STEPS
    
    # 2. Define the clients
    task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.6, 'wind': 0.5},
        {'label': 'Client_2_Earth', 'gravity': -9.8, 'wind': 0.5},
        {'label': 'Client_3_Mars', 'gravity': -3.73, 'wind': 0.5},
    ]
    single_task_list = list(task_list)
    
    add_derived_tasks(task_list, num_to_add_per_task=10)
    clients_per_round = int(len(task_list)/2)
    pprint.pprint(task_list)
    
    # Copy for single agent
    

    n_envs = 4

    # ==========================================
    # EXPERIMENT 2: Standard FL
    # ==========================================
    print("\n>>> Starting FL Experiment...")
    start_time = time.perf_counter()
    try:
        run_fl_experiment(
            NUM_ROUNDS=NUM_ROUNDS, 
            CHECK_FREQ=CHECK_FREQ, 
            LOCAL_STEPS=LOCAL_STEPS, 
            task_list=task_list,
            clients_per_round=clients_per_round,
            n_envs=n_envs
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "FL", "Time": elapsed_time})
        print(f">>> FL Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! FL Failed: {e}")
        execution_times.append({"Function": "FL", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # EXPERIMENT 3: DP FL (Epsilon 30)
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=30)...")
    start_time = time.perf_counter()
    try:
        run_dp_fl_experiment(
            NUM_ROUNDS=NUM_ROUNDS, 
            CHECK_FREQ=CHECK_FREQ, 
            LOCAL_STEPS=LOCAL_STEPS, 
            task_list=task_list, 
            DP_SENSITIVITY=15.0, 
            DP_EPSILON=30.0,
            clients_per_round=clients_per_round,
            n_envs=n_envs
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 1", "Time": elapsed_time})
        print(f">>> DP FL 1 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 1 Failed: {e}")
        execution_times.append({"Function": "DP FL 1", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # EXPERIMENT 4: DP FL (Epsilon 10)
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=10)...")
    start_time = time.perf_counter()
    try:
        run_dp_fl_experiment(
            NUM_ROUNDS=NUM_ROUNDS, 
            CHECK_FREQ=CHECK_FREQ, 
            LOCAL_STEPS=LOCAL_STEPS, 
            task_list=task_list, 
            DP_SENSITIVITY=15.0, 
            DP_EPSILON=10.0,
            clients_per_round=clients_per_round,
            n_envs=n_envs
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 2", "Time": elapsed_time})
        print(f">>> DP FL 2 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 2 Failed: {e}")
        execution_times.append({"Function": "DP FL 2", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # EXPERIMENT 5: DP FL (Epsilon 5)
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=5)...")
    start_time = time.perf_counter()
    try:
        run_dp_fl_experiment(
            NUM_ROUNDS=NUM_ROUNDS, 
            CHECK_FREQ=CHECK_FREQ, 
            LOCAL_STEPS=LOCAL_STEPS, 
            task_list=task_list, 
            DP_SENSITIVITY=15.0, 
            DP_EPSILON=5.0,
            clients_per_round=clients_per_round,
            n_envs=n_envs
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 3", "Time": elapsed_time})
        print(f">>> DP FL 3 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 3 Failed: {e}")
        execution_times.append({"Function": "DP FL 3", "Time": 0.0, "Error": str(e)})

        # ==========================================
    # EXPERIMENT 1: Single Agent
    # ==========================================
    print("\n>>> Starting Single Agent Experiment...")
    start_time = time.perf_counter()
    try:
        run_experiment_training(
            TOTAL_TIMESTEPS=total_steps,
            CHECK_FREQ=CHECK_FREQ, 
            task_list=single_task_list,
            n_envs=n_envs    
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "Single Agent", "Time": elapsed_time})
        print(f">>> Single Agent Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! Single Agent Failed: {e}")
        execution_times.append({"Function": "Single Agent", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # LOGGING
    # ==========================================
    print("\nSaving execution times to file...")
    with open("execution_times.txt", "w") as f:
        for entry in execution_times:
            # Used .get() with standard keys I defined above ("Function" and "Time")
            func_name = entry.get('Function', 'Unknown') 
            time_val = entry.get('Time', 0.0)
            err_msg = entry.get('Error', '')
            
            if err_msg:
                f.write(f"Function: {func_name}, FAILED, Error: {err_msg}\n")
            else:
                f.write(f"Function: {func_name}, Time Taken: {time_val:.6f} seconds\n")