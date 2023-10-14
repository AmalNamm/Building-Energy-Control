#from citylearn.agents.rbc import BasicRBC
#from citylearn.agents.rlc import RLC
from citylearn.agents.sac import SAC
#from citylearn.agents.q_learning import TabularQLearning

#class BasicRBCAgent(BasicRBC):
#class BasicRBCAgent(RLC):
#class BasicRBCAgent(SACRBC):
class BasicRBCAgent(SAC):
    kwargs = {
    'learning_rate': 0.0003,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,}
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations) 
    #hello