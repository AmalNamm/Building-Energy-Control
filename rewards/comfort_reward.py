#from citylearn.reward_function import ComfortReward
from citylearn.reward_function import IndependentSACReward
#class ComfortRewardFunction(ComfortReward):
class ComfortRewardFunction(IndependentSACReward):

    """ Simple passthrough example of comfort reward from Citylearn env """
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
    
    def calculate(self, observations):
        return super().calculate(observations)
    
    
