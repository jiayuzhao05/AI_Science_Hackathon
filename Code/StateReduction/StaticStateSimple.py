import numpy as np

class StaticStateSimple():
    """
    This is the state function used per default. For better performance, replace with your own state function
    """
    def __init__(self,state_dim):
        """
        You can give the object parameters here. 
        state_dim       (int) number of dimensions you want your state space to have
        """
        self.state_dim = state_dim
        
    def fit(self):
        """
        This function is executed when creating a state characterization. Used only in dynamic state functions.
        """
        pass
    
    def get_state(self,response):
        """
        This state function simply encodes the first n spikes and their relative timing.
        """
        state    = np.zeros(self.state_dim)
        
        if response.shape[0] == 0:
            return state
        
        state[0] = response[0,1]/2-1 # To normalize
        for i in range(1,min(response.shape[0],self.state_dim-1)):
            if ((response[i-1,1] - response[i,1]) % 2) == 0:
                state[i] = 0
            elif (response[i-1,1] - response[i,1]) % 4 == 1:
                diff     = np.sqrt(response[i,0] - response[i-1,0])/5
                state[i] = diff
            else:
                diff     = np.sqrt(response[i,0] - response[i-1,0])/5
                state[i] = -diff
        
        return state
    
