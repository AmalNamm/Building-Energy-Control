import os
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.sac import SAC
from rewards.user_reward import SubmissionReward
from agents.user_agent import SubmissionAgent

import numpy as np
import time
import pandas as pd

class WrapperEnv:
    """
    Env to wrap provide Citylearn Env data without providing full env
    Preventing attribute access outside of the available functions
    """
    def __init__(self, env_data):
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.time_steps = env_data['time_steps']
        self.seconds_per_time_step = env_data['seconds_per_time_step']
        self.random_seed = env_data['random_seed']
        self.buildings_metadata = env_data['buildings_metadata']
        self.episode_tracker = env_data['episode_tracker']
    
    def get_metadata(self):
        return {'buildings': self.buildings_metadata}

def create_citylearn_env(config, reward_function):
    env = CityLearnEnv(config.SCHEMA, reward_function=reward_function,central_agent=False)

    env_data = dict(
        observation_names = env.observation_names,
        action_names = env.action_names,
        observation_space = env.observation_space,
        action_space = env.action_space,
        time_steps = env.time_steps,
        random_seed = 1234,
        episode_tracker = None,
        seconds_per_time_step = None,
        buildings_metadata = env.get_metadata()['buildings']
    )

    wrapper_env = WrapperEnv(env_data)
    return env, wrapper_env

def update_power_outage_random_seed(env: CityLearnEnv, random_seed: int) -> CityLearnEnv:
    """Update random seed used in generating power outage signals.
    
    Used to optionally update random seed for stochastic power outage model in all buildings.
    Random seeds should be updated before calling :py:meth:`citylearn.citylearn.CityLearnEnv.reset`.
    """

    for b in env.buildings:
        b.stochastic_power_outage_model.random_seed = random_seed

    return env




def evaluate_trained_model(model_path, config):
    # Create the environment
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)

    # Load the trained SAC agent
    #agent = SAC(env)  # Initialize with the environment
    agent = SubmissionAgent(wrapper_env)
    checkpoint = torch.load(model_path)
    agent.policy_net[0].load_state_dict(checkpoint['model_state_dict'])
    
    # Ensure the agent is in evaluation mode (important if the agent has components like BatchNorm or Dropout)
    agent.policy_net[0].eval()

    total_reward = 0
    done = False
    observations = env.reset()

    while not done:
        # Use the agent to select an action
        with torch.no_grad():  # Ensure no gradients are computed during evaluation
            actions = agent.predict(observations)
        
        # Step the environment
        observations, reward, done, _ = env.step(actions)
        #total_reward += reward
        #observations, reward, done, _ = env.step(actions)
        reward = sum(reward) if isinstance(reward, list) else reward
        total_reward += reward


    # Compute and print metrics after evaluation
    metrics_df = env.evaluate_citylearn_challenge()
    print("Evaluation Metrics:", metrics_df)
    print(f"Total reward obtained: {total_reward}")

    return metrics_df, total_reward

if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
    
    config = Config()
    model_path = "final_model.pt"  # The path to your saved model
    evaluate_trained_model(model_path, config)
    

