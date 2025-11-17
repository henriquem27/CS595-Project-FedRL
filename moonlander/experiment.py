from fl_moon import run_fl_experiment
from single_moon import run_experiment_training
from dp_moon import run_dp_fl_experiment  
import pprint
from clustered import run_fl_experiment_clustered

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
            # Skip this item if its label isn't in the expected format
            print(f"Skipping task with unexpected label: {task.get('label')}")
            continue

        # Create the specified number of new tasks
        # i will be 1, 2, 3...
        for i in range(1, num_to_add_per_task + 1):

            # Create a fresh copy for this new item
            new_task = new_task_base.copy()

            # Calculate the new ID by subtracting i
            # 1st new task: current_id - 1
            # 2nd new task: current_id - 2
            new_id = current_id - i

            # Assemble the new label
            new_task['label'] = f"{base_name}_{new_id}_{suffix}"
            new_task['wind'] = 0.5 + 1.5 * i  # Example: increase wind for each new task
            # Add the newly created task to our temporary list
            new_items.append(new_task)

    # After iterating through all original tasks,
    # add all the new items to the main list.
    original_list.extend(new_items)
    print(f"Added {len(new_items)} new tasks.")


if __name__ == "__main__":
    print("Running experiment.py as main script...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 3
    LOCAL_STEPS = 50000  # Steps *per round*
    CHECK_FREQ = 2000    # Log weights every 2000 steps
    total_steps = NUM_ROUNDS * LOCAL_STEPS
    # 2. Define the clients
    # This single list defines all your clients.
    task_list = [
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
    add_derived_tasks(task_list, num_to_add_per_task=5)

    pprint.pprint(task_list)
    # single
    run_experiment_training(TOTAL_TIMESTEPS=total_steps,CHECK_FREQ=CHECK_FREQ, task_list=task_list)

    
    

    add_derived_tasks(task_list, num_to_add_per_task=5)

    

    # 3. Run the experiment
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list)

    run_fl_experiment_clustered(
        NUM_ROUNDS,
        CHECK_FREQ,
        LOCAL_STEPS,
        task_list,
        N_CLUSTERS=3
    )

    run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, task_list, DP_SENSITIVITY=150.0, DP_EPSILON=300.0)