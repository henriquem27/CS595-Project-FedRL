from fl_moon import run_fl_experiment
from single_moon import run_experiment_training

if __name__ == "__main__":
    print("Running experiment.py as main script...")

    # 1. Define FL Hyperparameters
    NUM_ROUNDS = 5
    LOCAL_STEPS = 10000  # Steps *per round*
    CHECK_FREQ = 2000    # Log weights every 2000 steps
    total_steps = NUM_ROUNDS * LOCAL_STEPS
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

    run_experiment_training(TOTAL_TIMESTEPS=total_steps, CHECK_FREQ=CHECK_FREQ, task_list=fl_task_list)