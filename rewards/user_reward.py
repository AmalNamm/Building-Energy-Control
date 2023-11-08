#from rewards.comfort_reward import ComfortRewardFunction
#from rewards.comfort_reward import SolarPenaltyAndComfortReward
#from rewards.CustomReward import RewardFunction
#from rewards.KPIsReward import RewardFunction
#from rewards.reward_function import IndependentSACReward
#from rewards.reward_function import MARL
from rewards.MyReward import CombinedRewardFunction


###################################################################
#####                Specify your reward here                 #####
###################################################################

#SubmissionReward = ComfortRewardFunction

#SubmissionReward = SolarPenaltyAndComfortReward

#SubmissionReward = RewardFunction

#SubmissionReward = IndependentSACReward
#SubmissionReward = MARL
SubmissionReward = CombinedRewardFunction #MarlSolar