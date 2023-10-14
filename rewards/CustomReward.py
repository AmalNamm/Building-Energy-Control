from typing import Any, List, Mapping, Tuple, Union
import numpy as np
from citylearn.data import ZERO_DIVISION_PLACEHOLDER

class RewardFunction:
    r"""Base and default reward function class.

    The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        self.env_metadata = env_metadata

    @property
    def env_metadata(self) -> Mapping[str, Any]:
        """General static information about the environment."""

        return self.__env_metadata
    
    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all buildings."""

        return self.env_metadata['central_agent']
    
    @env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        self.__env_metadata = env_metadata

    #def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        #net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]

        #if self.central_agent:
        #    reward = [min(sum(net_electricity_consumption)*-1, 0.0)]
       # else:
       #     reward = [min(v*-1, 0.0) for v in net_electricity_consumption]

        #return reward

    
    
    
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates a multi-objective reward function.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at the current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step`.

        Returns
        -------
        reward: List[float]
            Multi-objective reward for the transition to the current timestep.
        """

        # Extract relevant information from observations
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        user_comfort = [o['indoor_dry_bulb_temperature'] for o in observations]  # Use indoor temperature as a proxy for user comfort
        cost = [o['electricity_pricing'] for o in observations]  # You should replace 'cost' with the actual cost observation.
        #cost = [sum(o['electricity_pricing_predicted_24h']) for o in observations]

        # Define weights for each objective
        weight_energy_consumption = 0.25
        weight_user_comfort = 0.6
        weight_cost = 0.3

        # Calculate individual objective values
        energy_consumption_objective = sum(net_electricity_consumption)
        user_comfort_objective = -sum(user_comfort)  # Minimize indoor temperature to improve comfort
        cost_objective = -sum(cost)  # Minimize cost

        # Combine objectives into a multi-objective reward using weighted sum
        multi_objective_reward = (
            weight_energy_consumption * energy_consumption_objective +
            weight_user_comfort * user_comfort_objective +
            weight_cost * cost_objective
        )

        # Scale the multi-objective reward to the range [-1, 1]
        scaled_reward = min(max(multi_objective_reward, -1.0), 1.0)

        return [scaled_reward] * len(observations)
