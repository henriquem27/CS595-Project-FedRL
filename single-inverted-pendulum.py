import gymnasium as gym

from stable_baselines3 import A2C,PPO


env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1,render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1,n_epochs=50,learning_rate=0.0003)
model.learn(total_timesteps=50000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()