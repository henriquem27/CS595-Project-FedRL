import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

def test_env_creation():
    print("Testing environment creation...")
    env_kwargs = {'gravity': -9.8, 'enable_wind': True, 'wind_power': 5.0}
    try:
        # Test single env
        env = gym.make("LunarLander-v3", **env_kwargs)
        print("Single env created successfully.")
        env.close()
    except Exception as e:
        print(f"Failed to create single env: {e}")

    try:
        # Test VecEnv
        print("Testing VecEnv creation...")
        env = make_vec_env("LunarLander-v3", n_envs=4, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
        print("VecEnv created successfully.")
        
        # Test PPO set_env
        print("Testing PPO set_env...")
        # Create dummy env for init
        dummy_env = gym.make("LunarLander-v3")
        model = PPO("MlpPolicy", dummy_env, verbose=0)
        dummy_env.close()
        model.set_env(None)
        print("Model env set to None.")
        
        model.set_env(env)
        print(f"Model env set to: {model.env}")
        
        if model.env is None:
            print("ERROR: Model env is still None!")
        else:
            print("Model env set successfully.")
            
        env.close()
        
    except Exception as e:
        print(f"Failed to create VecEnv or set env: {e}")

if __name__ == "__main__":
    test_env_creation()
