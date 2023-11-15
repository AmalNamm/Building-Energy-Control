
from typing import List, Tuple, Dict
import numpy as np
from citylearn.base import Environment

class PowerOutage:
    """
    Base stochastic power outage model class.
    """
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed

    @property
    def random_seed(self) -> int:
        return np.random.randint(0, 10000) if self.__random_seed is None else self.__random_seed

    @random_seed.setter
    def random_seed(self, value: int):
        self.__random_seed = value

    def get_signals(self, time_steps: int, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method") 

class DistributedOutagesPowerOutage(PowerOutage):
    """
    Power outage signal stochastic model allowing variable outage scenarios based on distributions for SAIFI and CAIDI.
    """
    def __init__(self, saifi_distribution: Dict[str, any] = {'type': 'uniform', 'params': (0.5, 2.0)},
                 caidi_distribution: Dict[str, any] = {'type': 'uniform', 'params': (100.0, 500.0)},
                 start_time_steps: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.saifi_distribution = saifi_distribution
        self.caidi_distribution = caidi_distribution
        self.start_time_steps = start_time_steps

    def get_value_from_distribution(self, distribution):
        dist_type = distribution['type']
        params = distribution['params']

        if dist_type == 'uniform':
            return np.random.uniform(*params)
        elif dist_type == 'normal':
            return np.random.normal(*params)
        elif dist_type == 'exponential':
            return np.random.exponential(*params)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def get_signals(self, time_steps: int, seconds_per_time_step: float, **kwargs) -> np.ndarray:
        nprs = np.random.RandomState(self.random_seed)
        days_per_year = 365.0
        seconds_per_day = 86400.0
        seconds_per_minute = 60.0
        time_steps_per_day = seconds_per_day / seconds_per_time_step
        time_steps_per_minute = seconds_per_minute / seconds_per_time_step
        day_count = time_steps / time_steps_per_day
        daily_outage_probability = self.saifi / days_per_year

        outage_days = nprs.binomial(n=1, p=daily_outage_probability, size=int(day_count))
        outage_day_ixs = np.nonzero(outage_days)[0]
        outage_day_count = len(outage_day_ixs)
        start_time_steps = list(range(int(time_steps_per_day))) if self.start_time_steps is None else self.start_time_steps
        outage_start_time_steps = nprs.choice(start_time_steps, size=outage_day_count)
        outage_durations = nprs.exponential(scale=self.caidi, size=outage_day_count)
        outage_duration_time_steps = (outage_durations * time_steps_per_minute).astype(int)

        signals = np.zeros(time_steps, dtype=int)
        for i, start_time_step, duration_time_steps in zip(outage_day_ixs, outage_start_time_steps, outage_duration_time_steps):
            start_ix = int(i * time_steps_per_day + start_time_step)
            end_ix = min(int(start_ix + duration_time_steps), time_steps)
            signals[start_ix:end_ix] = 1

        return signals

    def generate_scenarios(self, num_scenarios: int, time_steps: int, seconds_per_time_step: float):
        scenarios = []
        for _ in range(num_scenarios):
            self.saifi = self.get_value_from_distribution(self.saifi_distribution)
            self.caidi = self.get_value_from_distribution(self.caidi_distribution)
            scenario_signals = self.get_signals(time_steps, seconds_per_time_step)
            scenarios.append(scenario_signals)
        return scenarios