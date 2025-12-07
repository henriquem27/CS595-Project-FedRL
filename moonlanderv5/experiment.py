from fl_moon import run_fl_experiment
from dp_moon import run_dp_fl_experiment
from single_moon import run_experiment_training 
import random
import time
import multiprocessing
import torch # Assuming you use PyTorch
import gc   
import pprint
import os
import json

def run_isolated(   target_func, kwargs):
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
            base_name = parts[0]  
            current_id = int(parts[1])  
            suffix = parts[2]  
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
    print("Running experiment.py sequentially (Process Isolated)...")
    execution_times = []
    
    NUM_ROUNDS = 125
    LOCAL_STEPS = 10000
    CHECK_FREQ = 2000    
    TOTAL_TIMESTEPS = NUM_ROUNDS * LOCAL_STEPS
    

    single_task_list = [
        {'label': 'Client_1_Moon', 'gravity': -1.6, 'wind': 0.5},
        {'label': 'Client_2_Earth', 'gravity': -9.8, 'wind': 0.5},
        {'label': 'Client_3_Mars', 'gravity': -3.73, 'wind': 0.5},
    ]

    task_list = single_task_list.copy()
    
    add_derived_tasks(task_list, num_to_add_per_task=6)
    clients_per_round = int(len(task_list)/2)
    pprint.pprint(task_list)

    print("\nSaving task_list to JSON file...")
    try:
        with open("task_list.json", "w") as f:
            json.dump(task_list, f, indent=4)
        print(">>> task_list saved to task_list.json")
    except Exception as e:
        print(f"!!! Failed to save task_list: {e}")





     #low privacy high fidelity 
    print("\n>>> Starting DP FL Experiment (Ep=30, Sens=10)...")
    start_time = time.perf_counter()
    try:
        run_isolated(
            target_func=run_dp_fl_experiment,
            kwargs={
                'NUM_ROUNDS': NUM_ROUNDS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'LOCAL_STEPS': LOCAL_STEPS, 
                'task_list': task_list, 
                'DP_SENSITIVITY': 0.2, 
                'DP_EPSILON': 500,
                'clients_per_round': clients_per_round
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 2", "Time": elapsed_time})
        print(f">>> DP FL 2 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 2 Failed: {e}")
        execution_times.append({"Function": "DP FL 2", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # light privacu
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=30, Sens=5)...")
    start_time = time.perf_counter()
    try:
        run_isolated(
            target_func=run_dp_fl_experiment,
            kwargs={
                'NUM_ROUNDS': NUM_ROUNDS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'LOCAL_STEPS': LOCAL_STEPS, 
                'task_list': task_list, 
                'DP_SENSITIVITY': 0.2, 
                'DP_EPSILON': 100.0,
                'clients_per_round': clients_per_round
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 3", "Time": elapsed_time})
        print(f">>> DP FL 3 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 3 Failed: {e}")
        execution_times.append({"Function": "DP FL 3", "Time": 0.0, "Error": str(e)})

    # ==========================================
    # EXPERIMENT 4: sweet spot?
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=30, Sens=1)...")
    start_time = time.perf_counter()
    try:
        run_isolated(
            target_func=run_dp_fl_experiment,
            kwargs={
                'NUM_ROUNDS': NUM_ROUNDS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'LOCAL_STEPS': LOCAL_STEPS, 
                'task_list': task_list, 
                'DP_SENSITIVITY': 0.2, 
                'DP_EPSILON': 40.0,
                'clients_per_round': clients_per_round
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 4", "Time": elapsed_time})
        print(f">>> DP FL 4 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 4 Failed: {e}")
        execution_times.append({"Function": "DP FL 4", "Time": 0.0, "Error": str(e)})
    
    # ==========================================
    # EXPERIMENT 5: High Privacy
    # ==========================================
    print("\n>>> Starting DP FL Experiment (Ep=30, Sens=1)...")
    start_time = time.perf_counter()
    try:
        run_isolated(
            target_func=run_dp_fl_experiment,
            kwargs={
                'NUM_ROUNDS': NUM_ROUNDS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'LOCAL_STEPS': LOCAL_STEPS, 
                'task_list': task_list, 
                'DP_SENSITIVITY': 0.2, 
                'DP_EPSILON': 10.0,
                'clients_per_round': clients_per_round
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "DP FL 4", "Time": elapsed_time})
        print(f">>> DP FL 4 Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! DP FL 4 Failed: {e}")
        execution_times.append({"Function": "DP FL 4", "Time": 0.0, "Error": str(e)})
    
    # ==========================================
    # FL
    # ==========================================
    print("\n>>> Starting FL Experiment...")
    start_time = time.perf_counter()
    try:
        # WRAPPED IN ISOLATED PROCESS
        run_isolated(
            target_func=run_fl_experiment,
            kwargs={
                'NUM_ROUNDS': NUM_ROUNDS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'LOCAL_STEPS': LOCAL_STEPS, 
                'task_list': task_list,
                'clients_per_round': clients_per_round
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "FL", "Time": elapsed_time})
        print(f">>> FL Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! FL Failed: {e}")
        execution_times.append({"Function": "FL", "Time": 0.0, "Error": str(e)})


    print("\n>>> Starting Single Moon Experiment...")
    start_time = time.perf_counter()
    try:
        run_isolated(
            target_func=run_experiment_training,
            kwargs={
                'TOTAL_TIMESTEPS': TOTAL_TIMESTEPS, 
                'CHECK_FREQ': CHECK_FREQ, 
                'task_list': single_task_list, 
                'experiment_name': "single_moon"
            }
        )
        elapsed_time = time.perf_counter() - start_time
        execution_times.append({"Function": "Single Moon", "Time": elapsed_time})
        print(f">>> Single Moon Finished in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"!!! Single Moon Failed: {e}")
        execution_times.append({"Function": "Single Moon", "Time": 0.0, "Error": str(e)})
    
    
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
