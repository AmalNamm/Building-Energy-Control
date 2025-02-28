import numpy as np
import time
import os
import pandas as pd
from citylearn.citylearn import CityLearnEnv
import torch
from citylearn.agents.sac import SAC as RLAgent

"""
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward

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

def evaluate(config):
    print("Starting local evaluation")
    
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)
    print("Env Created")
    print('Current time step:', env.time_step)
    print('environment number of time steps:', env.time_steps)
    print('environment uses central agent:', env.central_agent)
    print('Common (shared) observations amogst buildings:', env.shared_observations)
    print('Number of buildings:', len(env.buildings))

    #kwargs = {
    #'learning_rate': 0.0003,
    #'buffer_size': 1000000,
    #'learning_starts': 100,
    #'batch_size': 256,
    #'tau': 0.005,
    #'gamma': 0.99,
    #'train_freq': 1,}
    
    #agent = SubmissionAgent(wrapper_env,**kwargs)
    #agent = SubmissionAgent(wrapper_env)
    #agent = SubmissionAgent(wrapper_env, model_path="final_model.pt")
    #agent = SubmissionAgent(env, model_path="final_model.pt")
    #model_path="final_model.pt"
    #checkpoint = torch.load(model_path)
    #agent.policy_net[0].load_state_dict(checkpoint['model_state_dict'])
    #agent.policy_net[0].eval()
    
    #agent = RLAgent(env)
    
    #agent.learn(episodes=2, deterministic_finish=True)
    #agent.learn(episodes=2, deterministic_finish=True)




    #observations = env.reset()
    #agent.predict(observations)
    metrics_df = env.evaluate_citylearn_challenge()
    print(metrics_df)



    #print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1
        
        
    
    config = Config()

    evaluate(config)
