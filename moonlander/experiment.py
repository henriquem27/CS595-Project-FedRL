from fl_moon import run_fl_experiment
from single_moon import run_experiment_training
from dp_moon import run_dp_fl_experiment  
import pprint

def add_derived_tasks(original_list, num_to_add_per_task):
    """
    Loops through an original list of tasks and generates new tasks.

    For each task in the original list, it creates `num_to_add_per_task`
    new versions, each time subtracting 1 more from the client ID.

    Args:
        original_list (list): The list of task dictionaries to modify.
        num_to_add_per_task (int): The number of new, derived tasks
                                     to create for each original task.
    """

    # A temporary list to hold all the newly generated tasks.
    # We do this to avoid an infinite loop by modifying the list
    # we are iterating over.
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

            # Add the newly created task to our temporary list
            new_items.append(new_task)

    # After iterating through all original tasks,
    # add all the new items to the main list.
    original_list.extend(new_items)
    print(f"Added {len(new_items)} new tasks.")


if __name__ == "__main__":
    print("Running experiment.py as main script...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 10
    LOCAL_STEPS = 10000  # Steps *per round*
    CHECK_FREQ = 2000    # Log weights every 2000 steps
    total_steps = NUM_ROUNDS * LOCAL_STEPS
    # 2. Define the clients
    # This single list defines all your clients.
    task_list = [
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
    # single
    run_experiment_training(TOTAL_TIMESTEPS=total_steps,CHECK_FREQ=CHECK_FREQ, task_list=task_list)

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

    add_derived_tasks(fl_task_list, num_to_add_per_task=10)

    pprint.pprint(fl_task_list)

    # 3. Run the experiment
    run_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, fl_task_list)

 

    run_dp_fl_experiment(NUM_ROUNDS, CHECK_FREQ, LOCAL_STEPS, fl_task_list, DP_SENSITIVITY=150.0, DP_EPSILON=300.0)