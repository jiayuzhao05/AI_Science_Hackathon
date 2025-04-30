import gymnasium as gym
from gymnasium import spaces

from Reward.LinearReward              import LinearReward      as Reward
from StateReduction.StaticStateSimple import StaticStateSimple as State

from HelperFunctions.check_action import check_action

import numpy as np

class SimulatedNetwork(gym.Env):
    """
    Create a time insensitive Simulated network environment, that can be interacted with just like the a real network. You can give this class up to 4 parameters:
    - action_dim       int               Number of stimuli used per applied stimulation pattern (i.e. action). Each stimulus is in {0,1,2,3,4}
    - state_dim        int               State dimension of state/observation space (reduced representation of neuronal activity)
    - reward_function  func  (optional)  Reward function that receives the neuronal activity as an input. If you do not set it, the default is being used.
    - state_function   func  (optional)  This function transforms your neuronal activity into a reduced representation in [-1,1]^n, where [-1,1] are numbers
                                         between -1 and +1 and n is the state_dim defined above.
    
    You can interact with the environment as with any other Gymnasium environment: https://gymnasium.farama.org/index.html
    
    It is advised to use this environment only to test if your algorithm can be executed, not for benchmarking the performance, as the simulated networks and
    biological networks have fundamentally different behavior.
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self,action_dim,state_dim,reward_object=None,state_object=None):
        super(SimulatedNetwork,self).__init__()

        self.action_dim = action_dim
        self.state_dim  =  state_dim
        
        if reward_object is None:
            reward_object = Reward()
        if state_object is None:
            state_object = State(self.state_dim)
        self.reward_object = reward_object
        self.state_object  =  state_object
        
        self.action_space      = spaces.MultiDiscrete((np.zeros((self.action_dim))+5).tolist())
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.state_dim,))

        # For initialization
        self.state   = np.zeros((self.state_dim,))
        self.reward  = 0
        self.stim_id = 0

    def reset(self,seed=None,options=None):
        """
        Reset the environment state.
        """
        
        super().reset(seed=seed)
        
        # Initialize the state by setting it to 0.
        self.state = np.zeros((self.state_dim,))
        return self.state,{}

    def step(self,action):
        """
        Apply action and return new state, reward, termination info, and extra info. This process is not time sensitive (i.e. waits for user).
        """

        # Check action:
        action, msg = check_action(action,self.action_dim)
        
        # Apply action and get response
        spikes = []
        elecs  = []
        for i in range(4):
            if np.random.random() < 0.1+0.05*i: # This is just to make the system assymetric
                spikes.append(np.random.random()*20)
                elecs.append(i)
        for i in range(self.action_dim):
            if action[i] == 0:
                continue
            elecs.append(action[i]-1)
            spikes.append(max(0,min(19.9999,(i+1)*4+np.random.randn())))
            elecs.append(action[i]%4)
            spikes.append(max(0,min(19.9999,(i+2)*4+np.random.randn())))
        if len(spikes) == 0:
            response = np.zeros((0,2))
        else:
            elecs    = np.array(elecs)
            spikes   = np.array(spikes)
            order    = np.argsort(spikes)
            spikes   = spikes[order]
            elecs    = elecs[order]
            response = np.stack([spikes,elecs],1)
            
        self.stim_id += 1
        
        # Define the space
        self.state  = self.state_object.get_state(response)
        
        # Calculate reward
        self.reward = self.reward_object.reward(response)

        # No truncation/termination
        terminated = False
        truncated  = False
        
        # Extra information to get information for the user
        info = {"spikes": spikes, 
                "elecs": elecs, 
                "action": action,
                "missed_cyc": 0, 
                "stim_id": self.stim_id, 
                "simulated": True,
                "comment": msg}

        return self.state,self.reward,terminated,truncated,info

    def render(self,mode="human"):
        """
        Rendering the the current state and reward.
        """
        
        print(f"Current state: {self.state}, Reward: {self.reward}")

    def close(self):
        """
        When closing environment. (Nothing needs be done)
        """
        
        pass
        
    def fit(self,*args,**kwargs):
        """
        This is used to fit the state function in case of a dynamic case.
        """
        
        self.state_object.fit(*args,**kwargs)
        
