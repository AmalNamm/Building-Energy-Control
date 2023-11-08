#from typing import List, Mapping, Union, Any

#ZERO_DIVISION_PLACEHOLDER = 0.0001  # This is a placeholder. Define it as per your logic.

#reward functions from citylearn github
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
        self.__lower_exponent = 2.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 3.0 if higher_exponent is None else higher_exponent


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        solar_rewards = []  # To store rewards from SolarPenaltyReward
        marl_rewards = []   # To store rewards from MARL
        comfort_rewards = [] # To store reward from Confort reward
        combined_rewards = []  # To store the combined rewards

        # Loop through each building to calculate the SolarPenaltyReward
        for o, m in zip(observations, self.env_metadata['buildings']):
            e = o['net_electricity_consumption']
            #cc = m['cooling_storage']['capacity'] #inactive action 
            #hc = m['heating_storage']['capacity'] #inactive action
            dc = m['dhw_storage']['capacity'] #active action
            print("dc = m['dhw_storage']['capacity']", dc)
            ec = m['electrical_storage']['capacity'] #active action
            print("ec = m['electrical_storage']['capacity']", ec)

            #cs = o.get('cooling_storage_soc', 0.0) #inactive observation
            #hs = o.get('heating_storage_soc', 0.0) #inactive observation
            ds = o.get('dhw_storage_soc', 0.0)  #"active": true, "shared_in_central_agent": false
            es = o.get('electrical_storage_soc', 0.0)  #"active": true, "shared_in_central_agent": false
            solar_reward = 0.0

            # Apply similar logic as in SolarPenaltyReward to calculate the solar reward
            #if cc > ZERO_DIVISION_PLACEHOLDER:
            #    solar_reward += -(1.0 + np.sign(e) * cs) * abs(e)
            #if hc > ZERO_DIVISION_PLACEHOLDER:
                #solar_reward += -(1.0 + np.sign(e) * hs) * abs(e)
            if dc > ZERO_DIVISION_PLACEHOLDER:
                solar_reward += -(1.0 + np.sign(e) * ds) * abs(e)*0.01 #scale down the electricity
            if ec > ZERO_DIVISION_PLACEHOLDER:
                solar_reward += -(1.0 + np.sign(e) * es) * abs(e)*0.01 #scale down the electricity

            solar_rewards.append(solar_reward)
            
            #Loop through each building to calculate the ConfortReward
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            lower_bound = set_point - self.band
            upper_bound = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)
            
            # Use the appropriate exponent based on whether the system is heating or cooling
            if indoor_dry_bulb_temperature < lower_bound:
                exponent = self.lower_exponent if heating else self.higher_exponent
                comfort_reward = -(delta**exponent)
            elif lower_bound <= indoor_dry_bulb_temperature < set_point:
                comfort_reward = 0.0 if heating else -delta
            elif set_point <= indoor_dry_bulb_temperature <= upper_bound:
                comfort_reward = -delta if heating else 0.0
            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                comfort_reward = -(delta**exponent)
                
            comfort_rewards.append(comfort_reward)

        # Calculate MARL components (district-level considerations)
        district_electricity_consumption = sum(o['net_electricity_consumption'] for o in observations)
        marl_reward_component = np.sign(district_electricity_consumption) * \
                                0.01 * district_electricity_consumption ** 2 * \
                                np.nanmax([0, district_electricity_consumption])

        # MARL rewards, considering individual buildings or a central agent controlling all buildings
        if self.central_agent:
            marl_rewards = [marl_reward_component]
        else:
            for e in [o['net_electricity_consumption'] for o in observations]:
                individual_marl_reward = np.sign(e) * 0.01 * e ** 2 * np.nanmax([0, district_electricity_consumption])
                marl_rewards.append(individual_marl_reward)

                
                
        # Combine the rewards for each building
        #for solar_reward, marl_reward in zip(solar_rewards, marl_rewards):
        #    combined_reward = 0.4*solar_reward + 0.6*marl_reward
        #    combined_rewards.append(combined_reward)
        
        # Combine the rewards for each building
        for solar_reward, marl_reward, comfort_reward in zip(solar_rewards, marl_rewards,comfort_rewards):
            #combined_reward = 0.3*solar_reward + 0.2*marl_reward +0.5*comfort_reward
            #combined_reward = 0.3*solar_reward + 0.1*marl_reward + 0.6*comfort_reward
            combined_reward = solar_reward + marl_reward + 0*comfort_reward
            combined_rewards.append(combined_reward)

        if self.central_agent:
            # If there's a central agent, return the sum of combined rewards
            return [sum(combined_rewards)]
        else:
            # Otherwise, return the list of combined rewards
            return combined_rewards
