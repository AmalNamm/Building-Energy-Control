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
    
    #def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        self.env_metadata = env_metadata
        #super().__init__(env_metadata)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent

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

    
    
    
    def calculate_KPI(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        
        #reward_list = []
        # Given observation, compute the individual KPI rewards
        

        # 1. Carbon Emissions
        carbon_intensity_ = np.array([o['carbon_intensity'] for o in observations])
        net_electricity_consumption_ = np.array([o['net_electricity_consumption'] for o in observations])
        carbon_emissions = carbon_intensity_*net_electricity_consumption_
        #carbon_emissions = [o['carbon_intensity'] for o in observations] * [o['net_electricity_consumption'] for o in observations]
        #carbon_emissions = observations['carbon_intensity'] * observations['net_electricity_consumption']
        max_carbon_emissions = 100 #1000  # Placeholder, you'd use an actual value from historical data or set a threshold
        normalized_carbon_emissions = carbon_emissions / max_carbon_emissions

        # 2. Discomfort Proportion (Unmet Hours)
        #discomfort = abs(observations['indoor_dry_bulb_temperature'] - observations['indoor_dry_bulb_temperature_set_point'])
        indoor_dry_bulb_temperature_ = np.array([o['indoor_dry_bulb_temperature'] for o in observations])
        indoor_dry_bulb_temperature_set_point_ = np.array([o['indoor_dry_bulb_temperature_set_point'] for o in observations])
        discomfort = abs(indoor_dry_bulb_temperature_-indoor_dry_bulb_temperature_set_point_)
        max_discomfort = 5  # Placeholder, this might be the max temperature difference you're willing to accept
        normalized_discomfort = discomfort / max_discomfort

        # 3. Ramping, Load Factor, Daily Peak, and All-time Peak
        # Placeholders for now; these will be more intricate and may require historical data
        ramping = 0.5  # Placeholder
        load_factor = 0.5  # Placeholder
        daily_peak = 0.5  # Placeholder
        annual_peak = 0.5  # Placeholder

        # 4. Thermal Resilience
        thermal_resilience = 1 - normalized_discomfort

        # 5. Unserved Energy
        #unserved_energy = observations['power_outage'] * observations['net_electricity_consumption']
        
        #max_unserved_energy = 1000  # Placeholder, you'd use an actual value from historical data or set a threshold
        #normalized_unserved_energy = unserved_energy / max_unserved_energy
        normalized_unserved_energy = 0.5

        # Compute the final reward using weights
        reward_ = (0.1 * normalized_carbon_emissions) - \
                 (0.3 * normalized_discomfort) + \
                 (0.075 * ramping) + \
                 (0.075 * load_factor) - \
                 (0.075 * daily_peak) - \
                 (0.075 * annual_peak) + \
                 (0.15 * thermal_resilience) - \
                 (0.15 * normalized_unserved_energy)

        return reward_
    
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        
        reward_list = []
        # Given observation, compute the individual KPI rewards
        
        for o, m in zip(observations, self.env_metadata['buildings']):
            
            reward = 0.0
            
            # 1. Carbon Emissions
            

            carbon_emissions = o['carbon_intensity'] * o['net_electricity_consumption']
            max_carbon_emissions =  0.556063 #1000  # Placeholder, you'd use an actual value from historical data or set a threshold
            normalized_carbon_emissions = carbon_emissions / max_carbon_emissions
            
            reward += 0.1 * normalized_carbon_emissions
            
            # 2. Discomfort Proportion (Unmet Hours)
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
                reward += -(delta**exponent)*0.3
                normalized_discomfort = reward

            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                reward += 0.0 if heating else (-delta + 1.0)*0.3  # +1.0 for being inside the comfort zone
                normalized_discomfort = reward

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = (-delta + 1.0)*0.3 if heating else 0.0 # +1.0 for being inside the comfort zone
                normalized_discomfort = reward

            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                reward += -(delta**exponent)*0.3
                normalized_discomfort = reward
                
            # 4. Thermal Resilience
            thermal_resilience = 0.15 *(1 - normalized_discomfort)
            reward += 0.15 * thermal_resilience
            
            #minimize electricity consumption and maximize solar generation to charge energy storage systems
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
            
            
            #reward = 0.0
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_PLACEHOLDER else 0.0
            
            # 4. Thermal Resilience
            thermal_resilience = 1 - normalized_discomfort
            
            # 3. Ramping, Load Factor, Daily Peak, and All-time Peak
        # Placeholders for now; these will be more intricate and may require historical data
            ramping = 0.5  # Placeholder
            load_factor = 0.5  # Placeholder
            daily_peak = 0.5  # Placeholder
            annual_peak = 0.5  # Placeholder
            
            reward = (0.1 * normalized_carbon_emissions) - \
                 (0.3 * normalized_discomfort) + \
                 (0.075 * ramping) + \
                 (0.075 * load_factor) - \
                 (0.075 * daily_peak) - \
                 (0.075 * annual_peak) + \
                 (0.15 * thermal_resilience) 
            #- \
                 # (0.15 * normalized_unserved_energy)

            
            reward_list.append(reward)
            
        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list
        
        return reward
        
        


        # 3. Ramping, Load Factor, Daily Peak, and All-time Peak
        # Placeholders for now; these will be more intricate and may require historical data
        #ramping = 0.5  # Placeholder
        #load_factor = 0.5  # Placeholder
        #daily_peak = 0.5  # Placeholder
        #annual_peak = 0.5  # Placeholder

        

        # 5. Unserved Energy
        #unserved_energy = observations['power_outage'] * observations['net_electricity_consumption']
        
        #max_unserved_energy = 1000  # Placeholder, you'd use an actual value from historical data or set a threshold
        #normalized_unserved_energy = unserved_energy / max_unserved_energy
        #normalized_unserved_energy = 0.5

        # Compute the final reward using weights
        #reward = (0.1 * normalized_carbon_emissions) - \
         #        (0.3 * normalized_discomfort) + \
          #       (0.075 * ramping) + \
          #       (0.075 * load_factor) - \
          #       (0.075 * daily_peak) - \
          #       (0.075 * annual_peak) + \
          #       (0.15 * thermal_resilience) - \
          #       (0.15 * normalized_unserved_energy)

        #return reward
    

