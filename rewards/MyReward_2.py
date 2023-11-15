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

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
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

        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]

        if self.central_agent:
            reward = [min(sum(net_electricity_consumption)*-1, 0.0)]
        else:
            reward = [min(v*-1, 0.0) for v in net_electricity_consumption]

        return reward

class CombinedRewardFunction(RewardFunction):
    """
    Combines the SolarPenaltyReward and MARL rewards to incentivize both
    individual building efficiency and district-wide coordination.
    """

    def __init__(self, env_metadata: Mapping[str, Any],band: float = None, lower_exponent: float = None, higher_exponent: float = None, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent
        
    @property
    def band(self) -> float:
        return self.__band
    
    @property
    def lower_exponent(self) -> float:
        return self.__lower_exponent
    
    @property
    def higher_exponent(self) -> float:
        return self.__higher_exponent
    
    @band.setter
    def band(self, band: float):
        self.__band = 2.0 if band is None else band

    @lower_exponent.setter
    def lower_exponent(self, lower_exponent: float):
        self.__lower_exponent = 3.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 4.0 if higher_exponent is None else higher_exponent


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        
        comfort_rewards = []  # To store rewards from Confort reward
        # Loop through each building to calculate the SolarPenaltyReward
        for o, m in zip(observations, self.env_metadata['buildings']):
            
            #Extracting the observations
            # Extract relevant pieces from the observation
            indoor_temp = o['indoor_dry_bulb_temperature']
            temp_set_point = o['indoor_dry_bulb_temperature_set_point']
            carbon_intensity = o['carbon_intensity']
            net_electricity_consumption = o['net_electricity_consumption']
            power_outage = o['power_outage']
            electricity_price = o['electricity_pricing']

    # Extract the state of charge of energy storage systems
            dhw_storage_soc = o['dhw_storage_soc']
            electrical_storage_soc = o['electrical_storage_soc']
            ec = m['electrical_storage']['capacity']
            dc = m['dhw_storage']['capacity'] #active action
            # Components of the reward function
            cooling = heating_demand < cooling_demand
            #Confort for cooling/heating
            cooling_demand = o.get('cooling_demand', 0.0)
            heating_demand = o.get('heating_demand', 0.0)
            lower_bound = set_point - self.band
            upper_bound = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)
            # Use the appropriate exponent based on whether the system is heating or cooling
            if indoor_dry_bulb_temperature < lower_bound:
                exponent = self.higher_exponent if outage else self.lower_exponent
                comfort_reward = -(delta**exponent)
            elif lower_bound <= indoor_dry_bulb_temperature < set_point:
                comfort_reward = delta**2 if outage else delta #Give positive reward and higher reward if outage
                #comfort_reward = delta

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound:
                comfort_reward = delta**2 if outage else delta #Give positive reward and higher reward if outage
                #comfort_reward = delta
            else:
                exponent = self.higher_exponent if outage else self.lower_exponent
                comfort_reward = -(delta**exponent)
                
            comfort_rewards.append(comfort_reward)
            
            #
            MAX_SOC = m['electrical_storage']['capacity']
            MAX_SOC_dhw = m['dhw_storage']['capacity'] 
            
            #Confort for dhw demand
            dhw_demand = o.get('dhw_demand', 0.0)
            
            

            
        
    ######## ADD 20% Battery for all timesteps ##########

def reward_function(observation, action, weight_dict):
    # Extract relevant pieces from the observation
    indoor_temp = observation['indoor_dry_bulb_temperature']
    temp_set_point = observation['indoor_dry_bulb_temperature_set_point']
    carbon_intensity = observation['carbon_intensity']
    net_electricity_consumption = observation['net_electricity_consumption']
    power_outage = observation['power_outage']
    electricity_price = observation['electricity_pricing']

    # Extract the state of charge of energy storage systems
    dhw_storage_soc = observation['dhw_storage_soc']
    electrical_storage_soc = observation['electrical_storage_soc']

    # Components of the reward function
    comfort_penalty = -abs(indoor_temp - temp_set_point)
    emission_penalty = -carbon_intensity * net_electricity_consumption
    cost_penalty = -electricity_price * net_electricity_consumption
    resilience_reward = (dhw_storage_soc + electrical_storage_soc) * power_outage

    # Weight each part of the reward
    reward = (weight_dict['comfort_weight'] * comfort_penalty +
              weight_dict['emission_weight'] * emission_penalty +
              weight_dict['cost_weight'] * cost_penalty +
              weight_dict['resilience_weight'] * resilience_reward)

    return reward
