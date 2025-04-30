import numpy as np
from sklearn.decomposition import PCA

class DynamicStatePCA():
    """
    This is the state function used per default. For better performance, replace with your own state function
    """
    def __init__(self,state_dim):
        """
        You can give the object parameters here. 
        state_dim       (int) number of dimensions you want your state space to have
        """
        self.state_dim = state_dim
        self.PCA       = PCA(n_components=self.state_dim)
        
        # Fit dummy data to initialize PCA
        self.PCA.fit(np.random.randn(max(4*20,self.state_dim),4*20))
        
    def fit(self,spikes,elecs):
        """
        This function is executed when creating a state characterization. Used only in dynamic state functions.
        """
        X = np.zeros((len(spikes),4*20))
        
        for i in range(X.shape[0]):
            if len(spikes[i]) == 0:
                continue
            for j in range(spikes[i].shape[0]):
                X[0,int(elecs[i][j]*20+spikes[i][j])] = 1
        return X
    
    def get_state(self,response):
        """
        This state function simply encodes the first n spikes and their relative timing.
        """
        X     = np.zeros(4*20)
        state = np.zeros((1,4*20))
        
        if response.shape[0] == 0:
            return self.PCA.transform(state)
        
        # 0 is time 1 is elec
        
        for i in range(response.shape[0]):
            state[0,int(response[i,1]*20+response[i,0])] = 1
        
        return self.PCA.transform(state)
    
