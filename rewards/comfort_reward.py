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
        r"""Calculates reward, scaling it to the range [-1, 1]. ( to help with training stability) 

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Scaled reward for transition to the current timestep.
        """

        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]

        if self.central_agent:
            total_consumption = sum(net_electricity_consumption)
            # Ensure the total consumption is not zero to avoid division by zero
            if total_consumption != 0:
                reward = [min(total_consumption * -1 / abs(total_consumption), 1.0)]
            else:
                reward = [0.0]
        else:
            reward = [min(v * -1 / abs(v), 1.0) for v in net_electricity_consumption]

        return reward

        
class ComfortRewardFunction(RewardFunction):
    """Reward for occupant thermal comfort satisfaction.

    The reward is the calculated as the negative delta between the setpoint and indoor dry-bulb temperature raised to some exponent
    if outside the comfort band. If within the comfort band, the reward is the negative delta when in cooling mode and temperature
    is below the setpoint or when in heating mode and temperature is above the setpoint. The reward is 0 if within the comfort band
    and above the setpoint in cooling mode or below the setpoint and in heating mode.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-).
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env_metadata)
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
        reward_list = []
        
        # Get the electricity consumption-based reward from the base class
        comfort_reward = super().calculate(observations)

        for o in observations:
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - self.band
            upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)
            
            if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                exponent = self.lower_exponent if heating else self.higher_exponent
                reward = -(delta**exponent)
            
            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                reward = 0.0 if heating else -delta + 1.0  # +1.0 for being inside the comfort zone

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = -delta + 1.0 if heating else 0.0 # +1.0 for being inside the comfort zone

            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                reward = -(delta**exponent)
                
            reward_list.append(reward)
        
        alpha = 0.29
                
        if self.central_agent:
            final_reward = [alpha * comfort_reward[0] + (1-alpha) * sum(reward_list)]
        else:
            final_reward = [alpha * cr + (1-alpha) * rl for cr, rl in zip(comfort_reward, reward_list)]
        return final_reward

        
            
class SolarPenaltyReward(RewardFunction):
    """The reward is designed to minimize electricity consumption and maximize solar generation to charge energy storage systems.

    The reward is calculated for each building, i and summed to provide the agent with a reward that is representative of all the
    building or buildings (in centralized case)it controls. It encourages net-zero energy use by penalizing grid load satisfaction 
    when there is energy in the enerygy storage systems as well as penalizing net export when the energy storage systems are not
    fully charged through the penalty term. There is neither penalty nor reward when the energy storage systems are fully charged
    during net export to the grid. Whereas, when the energy storage systems are charged to capacity and there is net import from the 
    grid the penalty is maximized.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        
        

        for o, m in zip(observations, self.env_metadata['buildings']):
            e = o['net_electricity_consumption'] #net_electricity_consumption: active observation
            cc = m['cooling_storage']['capacity'] #cooling_storage : active action
            hc = m['heating_storage']['capacity'] #heating_storage : non active action
            dc = m['dhw_storage']['capacity']    #dhw_storage : active action
            ec = m['electrical_storage']['capacity'] #electrical_storage : active action
            cs = o.get('cooling_storage_soc', 0.0) #cooling_storage_soc  : non active observation
            hs = o.get('heating_storage_soc', 0.0)  #heating_storage_soc : non active observation
            ds = o.get('dhw_storage_soc', 0.0) #dhw_storage_soc :active observation
            es = o.get('electrical_storage_soc', 0.0) #electrical_storage_soc :active observation
            
            solar_gen = o['solar_generation']
        
        
            # Penalize wasted solar energy
            if solar_gen > 0 and e > 0:
                reward -= solar_gen

            # Reward optimal use of solar energy
            if solar_gen > 0 and e <= 0:
                reward += solar_gen
            
            
            reward = 0.0
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list
        
        return reward
    
        
class SolarPenaltyAndComfortReward(RewardFunction):
    """Addition of :py:class:`citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward`.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-).
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 3.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    coefficients: Tuple, default = (1.0, 1.0)
        Coefficents for `citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward` values respectively.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None, coefficients: Tuple = None):
        self.__functions: List[RewardFunction] = [
            SolarPenaltyReward(env_metadata),
            ComfortRewardFunction(env_metadata, band=band, lower_exponent=lower_exponent, higher_exponent=higher_exponent)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        RewardFunction.env_metadata.fset(self, env_metadata)

        for f in self.__functions:
            f.env_metadata = self.env_metadata
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients
        
    def get_time_of_use_multiplier(self, current_time: int) -> float:
        """Return a price multiplier based on the current time."""
        # Simple example: 
        # Peak hours are between 18:00 to 22:00 with 1.5x price.
        # Off-peak hours are all other times with 0.8x price.
        #if 18 <= current_time < 22:
        #if 16 <= current_time < 19:
        if 16 <= current_time < 19:
            return 1.5
        else:
            return 0.8
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        # Assume one of the observations contains the current time (e.g., in hours).
        #current_time = observations[0].get('current_time', 12)  # Default to noon if not provided.
        #current_time = observations['hour'] #for o in observations

        current_time = observations[0].get('hour',16) 
        #print ("************current time************", current_time)
        time_of_use_multiplier = self.get_time_of_use_multiplier(current_time)

        reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
        
        # Apply the multiplier to the reward coming from SolarPenaltyReward
        # Assuming SolarPenaltyReward is the first in the list of functions
        reward[0] = reward[0] * time_of_use_multiplier
        #print("********reward[0]*******",reward[0])
        reward = reward * np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()
        print("********reward*******",reward)

        return reward

    #def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
    #    reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
    #    reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
    #    reward = reward.sum(axis=0).tolist()

    #    return reward