#from citylearn.agents.rbc import BasicRBC
#from citylearn.agents.rlc import RLC
from citylearn.agents.sac import SAC
import os
#from citylearn.agents.sac import SACRBC
#from citylearn.agents.q_learning import TabularQLearning
#from citylearn.agents.sac import SAC as RLAgent
import torch
from agents.SACmodel_2 import SAC_TGELU
#from SACmodel_2 import SAC_TGELU

#from agents.SACmodel import SAC_TGELU_WithoutTarget



#class BasicRBCAgent(BasicRBC):
#class BasicRBCAgent(RLC):
#class BasicRBCAgent(SACRBC):
#class BasicRBCAgent(SAC_TGELU):
#class BasicRBCAgent(SAC_TGELU_WithoutTarget):
#class BasicRBCAgent():
class BasicRBCAgent(SAC):


    #kwargs = {
    #'learning_rate': 0.0003,
    #'buffer_size': 1000000,
    #'learning_starts': 100,
    #'batch_size': 256,
    #'tau': 0.005,
    #'gamma': 0.99,
    #'train_freq': 1,}
    """ Can be any subclass of citylearn.agents.base.Agent """
    #def __init__(self, env, **kwargs):
    #    super().__init__(env, **kwargs)
        
    #def __init__(self, env, model_path="final_model_50.pt", **kwargs):
    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)
        #if model_path:
            #checkpoint = torch.load(model_path)
            
        #for ix in enumerate(self.env.buildings):
                
            #self.policy_net[ix].load_state_dict(checkpoint[f'runs/sac-default-False-independent_sac/et2rpsev/policy_net{ix}.pth'])
            #self.policy_net[ix].eval()
            #self.policy_net[0].load_state_dict(checkpoint['model_state_dict_50'])
            #self.policy_net[0].eval()

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)
    #def train(self, observations):
      #  model = RLAgent(env)
      #  return super().learn(episodes=1, deterministic_finish=True)
        

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations) 
    #hello