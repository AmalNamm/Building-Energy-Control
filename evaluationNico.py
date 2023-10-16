import time
import os
import argparse

import wandb

from citylearn.citylearn import CityLearnEnv
from utilities.utils import set_seed

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

def evaluate(config, algorithm_config):
    print("Starting local evaluation")
    
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)
    print("Env Created")
    print('Current time step:', env.time_step)
    print('environment number of time steps:', env.time_steps)
    print('environment uses central agent:', env.central_agent)
    print('Common (shared) observations amogst buildings:', env.shared_observations)
    print('Number of buildings:', len(env.buildings))

    agent = SubmissionAgent(wrapper_env, **algorithm_config)

    observations = env.reset()

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(observations)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while True:
            
            ### This is only a reference script provided to allow you 
            ### to do local evaluation. The evaluator **DOES NOT** 
            ### use this script for orchestrating the evaluations. 

            observations, _, done, _ = env.step(actions)
            if not done:
                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                episodes_completed += 1
                metrics_df = env.evaluate_citylearn_challenge()
                episode_metrics.append(metrics_df)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}", )
                
                logging_metrics = {}

                for metric in metrics_df.values():
                    
                    logging_metrics[metric['display_name']] = metric['value']

                wandb.log(logging_metrics)

                # Optional: Uncomment line below to update power outage random seed 
                # from what was initially defined in schema
                env = update_power_outage_random_seed(env, 90000)

                observations = env.reset()

                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= config.num_episodes:
                break

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path", type=str, default="runs/sac/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--qf_nns", type=int, default=128)
    parser.add_argument("--pi_nns", type=int, default=128)

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    algorithm_config = {
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "train_freq": args.train_freq,
        "policy_kwargs": {
            "net_arch": {
                "pi": [args.pi_nns, 2 * args.pi_nns],
                "qf": [args.qf_nns, 2 * args.qf_nns],
            }
        },
    }

     # Initialize WandB
    with wandb.init(
        project="ugrip-energy",
        config={
            "algorithm": "sac",
            **algorithm_config,
        },
        name="sac_" + str(int(time.time())), 
        sync_tensorboard=True,
    ) as run:

        # Set up logging directory
        logs_path = args.logs_path + run.id + "/"
        os.makedirs(logs_path, exist_ok=True)

        class Config:
            data_dir = './data/'
            SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
            num_episodes = 1
            
        config = Config()

        evaluate(config, algorithm_config)
