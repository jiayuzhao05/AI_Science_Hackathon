import numpy as np

class LinearReward:
    """
    This is the reward function used per default and also for final evaluation. 
    
    This is functions give +1 reward for each spike pair that is clockwise and occuring 
    within 5 ms and -1 reward for each spike pair that is counter-clockwise and occuring 
    within 5 ms. 
    """
    def __init__(self):
        """
        You can use this function to pass some variables
        """
        pass

    def reward(self,response):
        """
        This function is being called when a response is transformed into a reward. Do not rename.
        """
        r = 0
        for i in range(2,response.shape[0]):
            if response[i,0] - response[i-1,0] > 5:
                continue
            if (response[i,1] - response[i-1,1]) % 4 == 1:
                r += 1
            elif (response[i,1] - response[i-1,1]) % 4 == 3:
                r -= 1
        return r
    
