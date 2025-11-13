import gymnasium as gym
from stable_baselines3 import PPO
import time


"""
This script trains a PPO agent on the LunarLander-v3 environment.
It first trains the agent in a non-rendered environment for speed,
then visualizes the trained agent in a rendered environment.
Did this just for testing purposes.
"""
# --- PHASE 1: TRAINING (Fast, no graphics) ---

# 1. Create environment with render_mode=None (Runs as fast as CPU allows)
# We use the standard LunarLander settings.
train_env = gym.make("LunarLander-v3",
                     continuous=False,
                     gravity=-1.6,
                     enable_wind=False,
                     wind_power=15.0,
                     turbulence_power=1.5,
                     render_mode=None)

# 2. Create the model
model = PPO("MlpPolicy", train_env, verbose=1)

# 3. Train for longer (100k steps is a good start for LunarLander)
print("Starting training...")
model.learn(total_timesteps=100000)
print("Training finished.")

# 4. Save the model
model.save("ppo_lunarlander")

# 5. Close the training env to free up resources
train_env.close()

# --- PHASE 2: VISUALIZATION (Real-time, with graphics) ---

# 1. Create a NEW environment specifically for human viewing
eval_env = gym.make("LunarLander-v3",
                    continuous=False,
                    gravity=-10.0,
                    enable_wind=False,
                    wind_power=15.0,
                    turbulence_power=1.5,
                    render_mode="human")

# 2. Load the model we just trained (optional if model object still exists, but good practice)
loaded_model = PPO.load("ppo_lunarlander")

# 3. Run the visualization loop
print("Starting visualization...")
obs, _ = eval_env.reset()

# 1. Set the timer
start_time = time.time()
RUN_DURATION = 15  # Run for 15 seconds

# 2. Check time in the loop condition
while time.time() - start_time < RUN_DURATION:
    action, _states = loaded_model.predict(obs)
    obs, rewards, dones, truncated, info = eval_env.step(action)

    if dones or truncated:
        obs, _ = eval_env.reset()

# 3. CRITICAL: Close the window when time is up
print("Time limit reached.")
eval_env.close()
