from .rl import * # from .rl import * in production
from .r3 import R3 # from .r3 import R3 in production
import time
import os
import re
from stable_baselines3 import PPO, DQN  
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

def main():
    ti = time.time()
    
    #left, down, up, right
    #environment = Environment('CartPole-v0')
    #environment.render()
    
    env = R3(fitness_threshold=1000) 
    """print(env.observation_space.sample().shape)
    env.sow()
    print(env.layers.get_column('pipelines', 'sowing_dates').to_numpy().shape)"""
    
    #check_env(env)
    models_dir = "models/DQN"
    logdir = "logs"

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    
    EPISODES = 100
    TIMESTEPS = 100
    for episode in range(EPISODES):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
        model.save(f'{models_dir}/' + f'{TIMESTEPS*episode}')
    """obs = env.reset()
    x = []
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        x.append(reward)
        env.render()
        if done:
            obs = env.reset()

    print(x)"""
    env.close()
    
    
    print(time.time() - ti)


if __name__ == '__main__':
    main()