from single_agent_train import run_experiment_training
from dif_fl_train import run_dp_fl_experiment
from fl_train import run_fl_experiment
tasks_to_run = [
    {
        'label': 'StandardDoublePendulum',
        'mask': None  # This is your "Agent 1"
    },
    {
        'label': 'PenalizedDoublePendulum67',
        'mask': [6, 7]  # This is your "Agent 2"
    },
    {
        'label': 'PenalizedDoublePendulum34',
        'mask': [3, 4]  # This is your "Agent 3"
    },
    # {
    #     'label': 'PenalizedDoublePendulum123',
    #     'mask': [1, 2, 3]
    # }
]


run_experiment_training(
    TOTAL_TIMESTEPS=100000,
    CHECK_FREQ=1000,
    task_list=tasks_to_run
)
"""
|--- Hyperparameters for differential privacy ---|
"""
NUM_ROUNDS = 20
LOCAL_STEPS = 5000
TOTAL_TIMESTEPS = NUM_ROUNDS * LOCAL_STEPS
CHECK_FREQ = 5000  # Callback check frequency
DP_SENSITIVITY = 150.0
DP_EPSILON = 300.0

run_dp_fl_experiment(
    NUM_ROUNDS=NUM_ROUNDS,
    CHECK_FREQ=CHECK_FREQ,
    LOCAL_STEPS=LOCAL_STEPS,
    task_list=tasks_to_run,
    DP_SENSITIVITY=DP_SENSITIVITY,
    DP_EPSILON=DP_EPSILON
)
run_fl_experiment(
    NUM_ROUNDS=NUM_ROUNDS,
    CHECK_FREQ=CHECK_FREQ,
    LOCAL_STEPS=LOCAL_STEPS,
    task_list=tasks_to_run
)