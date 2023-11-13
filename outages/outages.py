from typing import List
import numpy as np
from citylearn.base import Environment

class PowerOutage:
    """Base stochastic power outage model class.
    
    Randomly assigns power outage signal to time steps.

    Parameters
    ----------
    random_seed: int, optional
        Pseudorandom number generator seed for repeatable results.
    """

    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed 

    @property
    def random_seed(self) -> int:
        return np.random.randint(*Environment.DEFAULT_RANDOM_SEED_RANGE) if self.__random_seed is None else self.__random_seed

    @random_seed.setter
    def random_seed(self, value: int):
        self.__random_seed = value

    def get_signals(self, time_steps: int, **kwargs) -> np.ndarray:
        """Returns power outage signal time series.

        Returns time series with randomly selected time steps set as candidates for power outage.

        Parameters
        ----------
        time_steps: int
            Number of time steps in returned signal time series.
        
        Other Parameters
        ----------------
        kwargs: Any
            Any other parameters specific to :py:meth:`get_signals` method for other subclasses of 
            :py:class:`citylearn.grid_resilience.PowerOutage`.
        
        Returns
        -------
        signals: np.ndarray
            Power outage time series signal where value of 0 indicates no power outage at time step index 
            whereas value of 1 indicates a power outage at said time step.
        """

        nprs = np.random.RandomState(self.random_seed)
        signals = nprs.choice([0, 1], size=time_steps) 

        return signals  

class ReliabilityMetricsPowerOutage(PowerOutage):
    """Power outage signal stochastic model based on Distribution System Reliability Metrics.

    Generates time series of power outage signals based on System Average Interruption Frequency Index (SAIFI)
    and Customer Average Interruption Duration Index (CAIDI). The signal is generated by sampling `n` instances 
    from a binomial distribution to select days that experience power outage where the probability, `p`, is the 
    ratio of `saifi` to number of days in a year (365). The start time step index for the outage on each day is 
    then randomly selected from a uniform distribution of `start_time_steps` or all valid daily time step indexes. 
    Finally the duration of each power outage event is set by sampling from an exponential distribution with a 
    scale set to `caidi`.
    
    Parameters
    ----------
    saifi: float, default: 1.436
        Number of non-momentary electric interruptions, per year, the average customer experienced 
        and is used as the average number of days per year that experience power outage.
    caidi: float, default: 331.2
        Average number of minutes it takes to restore non-momentary electric interruptions 
        and is used as the average length of a power outage.
    start_time_steps: List[int], optional
        List of candidate daily time step indexes to randomly select from when deciding start time step of an outage.
        For example, for an hourly simulation that wants to consider power outages that start during the evening peak 
        between 4 PM to 7 PM will set `start_time_steps` as [15, 16, 17, 18]. By default all daily time step indexes 
        are considered.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments used to initialize :py:class:`citylearn.grid_resilience.PowerOutage` super class.

    Notes
    -----
    The reliability metrics are sourced from https://www.eia.gov/electricity/annual/html/epa_11_01.html.
    """
    
    def __init__(self, saifi: float = None, caidi: float = None, start_time_steps: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.saifi = saifi
        self.caidi = caidi
        self.start_time_steps = start_time_steps

    @property
    def saifi(self) -> float:
        return self.__saifi
    
    @property
    def caidi(self) -> float:
        return self.__caidi
    
    @property
    def start_time_steps(self) -> List[int]:
        return self.__start_time_steps
    
    @saifi.setter
    def saifi(self, value: float):
        self.__saifi = 1.436 if value is None else value

    @caidi.setter
    def caidi(self, value: float):
        self.__caidi = 331.2 if value is None else value

    @start_time_steps.setter
    def start_time_steps(self, value: List[float]):
        self.__start_time_steps = value

    def get_signals(self, time_steps: int, seconds_per_time_step: float, **kwargs) -> np.ndarray:
        """Returns power outage signal time series.

        Returns time series with randomly selected time steps set as candidates for power outage.

        Parameters
        ----------
        time_steps: int
            Number of time steps in returned signal time series.
        seconds_per_time_step: float
            Number of seconds in one `time_step`.
        
        Other Parameters
        ----------------
        kwargs: Any
            Any other parameters specific to :py:meth:`get_signals` method for other subclasses of 
            :py:class:`citylearn.grid_resilience.PowerOutage`.
        
        Returns
        -------
        signals: np.ndarray
            Power outage time series signal where value of 0 indicates no power outage at time step index 
            whereas value of 1 indicates a power outage at said time step.
        """

        nprs = np.random.RandomState(self.random_seed)
        days_per_year = 365.0
        seconds_per_day = 86400.0
        seconds_per_minute = 60.0
        time_steps_per_day = seconds_per_day/seconds_per_time_step
        time_steps_per_minute = seconds_per_minute/seconds_per_time_step
        day_count = time_steps/time_steps_per_day
        daily_outage_probability = self.saifi/days_per_year #Determines the daily outage probability based on the self.saifi value and the number of days in a year.

        outage_days = nprs.binomial(n=1, p=daily_outage_probability, size=int(day_count)) #Generates a binary array (outage_days) indicating which days have outages, using a binomial distribution.
        outage_day_ixs = outage_days*np.arange(day_count)
        outage_day_ixs = outage_day_ixs[outage_day_ixs != 0]
        outage_day_count = outage_days[outage_days == 1].shape[0]
        start_time_steps = list(range(int(time_steps_per_day))) if self.start_time_steps is None else self.start_time_steps
        outage_start_time_steps = nprs.choice(start_time_steps, size=outage_day_count)
        outage_durations = nprs.exponential(scale=self.caidi, size=outage_day_count) # [mins] #Uses an exponential distribution to determine the duration of each outage, based on self.caidi.
        outage_duration_time_steps = outage_durations*time_steps_per_minute #Converts the outage duration from minutes to time steps.

        signals = np.zeros(time_steps, dtype=int)
        
#For each outage, it updates the relevant indices in signals to 1, indicating a power outage.
        for i, j , k in zip(outage_day_ixs, outage_start_time_steps, outage_duration_time_steps):
            start_ix = i*time_steps_per_day + j
            end_ix = start_ix + k
            start_ix = int(start_ix)
            end_ix = int(end_ix)
            signals[start_ix:end_ix] = 1
        
        return signals #The function returns an array (signals) representing the time series of power outages.