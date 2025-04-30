import numpy as np

class ExponentialReward:
    """
    This is functions give +n reward for each spike pair that is clockwise and occuring 
    within 5 ms and -n reward for each spike pair that is counter-clockwise and occuring 
    within 5 ms. n doubles each time two spike pairs have the same direction and are within
    5 ms, but resets to 1 in all other cases. 
    
    Compared to the linear reward, this reward function focuses on having spikes going in a circle.
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
        f = 0
        for i in range(2,response.shape[0]):
            if response[i,0] - response[i-1,0] > 5:
                f = 0
            if (response[i,1] - response[i-1,1]) % 4 == 1:
                if f <= 0:
                    f = 1
                else:
                    f *= 2
                r += f
            elif (response[i,1] - response[i-1,1]) % 4 == 3:
                if f >= 0:
                    f = -1
                else:
                    f *= 2
                r += f
        return r
