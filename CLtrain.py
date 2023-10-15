from citylearn.agents.sac import SAC as RLAgent
from citylearn.citylearn import CityLearnEnv, EvaluationCondition

dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(dataset_name, central_agent=False, simulation_end_time_step=1000)
model = RLAgent(env)
model.learn(episodes=2, deterministic_finish=True)

# print cost functions at the end of episode
kpis = model.env.evaluate(baseline_condition=EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV)
kpis = kpis.pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')



from agents.SACmodel import SAC
import random
import numpy as np
import time
import os
import pandas as pd
from citylearn.citylearn import CityLearnEnv
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
        random_seed = None,
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

def train_sac_with_hyperparameters(config, lr, tau, gamma):
    
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)

    
    # Initialize the SAC model with the given hyperparameters
    #sac_model = SAC(wrapper_env,lr=lr, tau=tau, gamma=gamma)
    
    model = RLAgent(env)
    model.learn(episodes=1, deterministic_finish=True)
    # Save the trained model
    #model.save("sac_model")
    #print(type(model))
    #print(dir(model))

    



#print(f"Best Hyperparameters: {best_hyperparameters} with reward: {best_reward}")



if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1
        
        
        
        
    
    config = Config()
    
    # Train SAC with selected hyperparameters
    lr = 0.01
    tau = 0.01
    gamma = 0.95
    reward = train_sac_with_hyperparameters(config,lr, tau, gamma)  # Define this function to train SAC and return reward

    #evaluate(config)