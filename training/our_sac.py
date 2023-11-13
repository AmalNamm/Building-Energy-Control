import logging
import time
import os
import argparse
import numpy as np

import wandb

from citylearn.citylearn import CityLearnEnv
from utilities.utils import set_seed

#from agents.our_sac_agent import OurSAC as SAC
#from rewards import IndependentSACReward, WeightedReward

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

def train(config, algorithm_config, episodes: int = None):
       
    """Train agent.

    Parameters
    ----------
    episodes: int, default: 1
        Number of training episode >= 1.
    logging_level: int, default: 30
        Logging level where increasing the number silences lower level information.
    """
    
    print("Starting local training")

    best_score = np.inf

    env, wrapper_env = create_citylearn_env(config, SubmissionReward)

    agent = SubmissionAgent(wrapper_env, **algorithm_config)

    episodes_completed = 0

    episodes = 1 if episodes is None else episodes

    for episode in range(episodes):

        observations = env.reset()
        
        episode_time_steps = env.time_steps
        done = False
        time_step = 0
        rewards_list = []

        while not done:
            actions = agent.predict(observations)

            # apply actions to citylearn_env
            next_observations, rewards, done, _ = env.step(actions)
            rewards_list.append(rewards)

            # update
            agent.update(observations, actions, rewards, next_observations, done=done)

            observations = [o for o in next_observations]

            logging.debug(
                f'Time step: {time_step + 1}/{episode_time_steps},'\
                    f' Episode: {episode + 1}/{episodes},'\
                        f' Actions: {actions},'\
                            f' Rewards: {rewards}'
            )

            time_step += 1

        rewards = np.array(rewards_list, dtype='float')
        rewards_summary = {
            'min': rewards.min(axis=0),
            'max': rewards.max(axis=0),
            'sum': rewards.sum(axis=0),
            'mean': rewards.mean(axis=0)
        }

        episodes_completed += 1

        # Log metrics

        metrics_df = env.evaluate_citylearn_challenge()
        print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}", )
        
        if wandb_enabled:

            logging_metrics = {}

            for metric in metrics_df.values():
                
                logging_metrics[metric['display_name'].replace(" ", "_").lower()] = metric['value']

            wandb.log(logging_metrics)

            if logging_metrics['score'] < best_score:

                best_score = logging_metrics['score']
                wandb.run.summary["best_score"] = best_score

                # Save model in the run folder

                agent.save_policy(path=logs_path)

        logging.info(f'Completed episode: {episode + 1}/{episodes}, Reward: {rewards_summary}')

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument("--reward", type=str, default="independent_sac")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--logs_path", type=str, default="runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy_type", type=str, default="default")
    parser.add_argument("--use_target", action="store_true")
    parser.add_argument("--p_hidden_dimension", type=int, default=256)
    parser.add_argument("--q_hidden_dimension", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_buffer_capacity", type=int, default=1e5)
    parser.add_argument("--standardize_start_time_step", type=int)
    parser.add_argument("--end_exploration_time_step", type=int)
    parser.add_argument("--action_scaling_coefficient", type=float, default=0.5)
    parser.add_argument("--reward_scaling", type=float, default=5.0)
    parser.add_argument("--update_per_time_step", type=int, default=2)
    parser.add_argument("--tgelu_tr", type=float, default=1)

    args = parser.parse_args()

    # Setup agent and reward according to parameters

    if args.agent == "sac":

        SubmissionAgent = SAC

    if args.reward == "independent_sac":

        SubmissionReward = IndependentSACReward

    elif args.reward == "weighted_reward":

        SubmissionReward = WeightedReward

    # Set seed for reproducibility
    set_seed(args.seed)

    algorithm_config = {
        "p_hidden_dimension": [args.p_hidden_dimension, 2 * args.p_hidden_dimension],
        "q_hidden_dimension": [args.q_hidden_dimension, 2 * args.q_hidden_dimension],
        "discount": args.discount,
        "tau": args.tau,
        "alpha": args.alpha,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "standardize_start_time_step": args.standardize_start_time_step,
        "end_exploration_time_step": args.end_exploration_time_step,
        "action_scaling_coefficient": args.action_scaling_coefficient,
        "reward_scaling": args.reward_scaling,
        "update_per_time_step": args.update_per_time_step,
        "policy_type": args.policy_type,
        "use_target": args.use_target,
        "tgelu_tr": args.tgelu_tr,
    }

     # Initialize WandB
    
    wandb_enabled = args.wandb

    experiment_name = f"{args.agent}-{args.policy_type}-{args.use_target}-{args.reward}"

    if wandb_enabled:

        run = wandb.init(
            project="cl-2023",
            entity="optimllab",
            config={
                "algorithm": experiment_name,
                **algorithm_config,
            },
            name=f"{experiment_name}_{int(time.time())}", 
            sync_tensorboard=True,
        )

        # Set up logging directory
        logs_path = f"{args.logs_path}/{experiment_name}/{run.id}/"
        os.makedirs(logs_path, exist_ok=True)

    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1
        
    config = Config()

    train(config, algorithm_config, episodes=args.epochs)
