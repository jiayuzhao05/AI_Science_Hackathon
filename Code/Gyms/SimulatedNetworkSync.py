import gymnasium as gym
from gymnasium import spaces

from Reward.LinearReward              import LinearReward      as Reward
from StateReduction.StaticStateSimple import StaticStateSimple as State

from HelperFunctions.check_action import check_action

import multiprocessing

import numpy as np
import time

class SimulatedNetworkSync(gym.Env):
    """
    Create a time sensitive Simulated network environment, that can be interacted with just like the a real network. The sync case - different from the base 
    case - will give you your stimulation results in a periodic fashion (just like the biological networks). You can give this class up to 5 parameters:
    - action_dim       int               Number of stimuli used per applied stimulation pattern (i.e. action). Each stimulus is in {0,1,2,3,4}
    - state_dim        int               State dimension of state/observation space (reduced representation of neuronal activity)
    - stim_period      float (optional)  The time between two applied actions/stimulus patterns. This number is 250 ms by default.
    - reward_function  func  (optional)  Reward function that receives the neuronal activity as an input. If you do not set it, the default is being used.
    - state_function   func  (optional)  This function transforms your neuronal activity into a reduced representation in [-1,1]^n, where [-1,1] are numbers
                                         between -1 and +1 and n is the state_dim defined above.
    
    You can interact with the environment as with any other Gymnasium environment: https://gymnasium.farama.org/index.html
    
    It is advised to use this environment only to test if your algorithm can be executed, not for benchmarking the performance, as the simulated networks and
    biological networks have fundamentally different behavior. 
    
    If you do not send an action in time, the default action is being used, which does not stimulate the network at all (all zeros).
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self,action_dim,state_dim,stim_period=250.,reward_object=None,state_object=None):
        super(SimulatedNetworkSync,self).__init__()

        self.action_dim  =  action_dim
        self.state_dim   =   state_dim
        self.stim_period = stim_period
        
        self.stimulus_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        
        if reward_object is None:
            reward_object = Reward()
        if state_object is None:
            state_object = State(self.state_dim)
        self.reward_object = reward_object
        self.state_object  =  state_object
        
        self.action_space      = spaces.MultiDiscrete((np.zeros((self.action_dim))+5).tolist())
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.state_dim,))

        # For initialization
        self.state          = np.zeros((self.state_dim,))
        self.reward         = 0
        self.missed_stimuli = 0
        
        # Starting the process for simulation
        self.process = multiprocessing.Process(target=self.process_function,args=(self.stimulus_queue,self.response_queue))
        self.process.start()
        
    def process_function(self,stimulus_queue,response_queue):
        """
        This function is running in another process, with the goal of stimulating the network periodically. Do not run this function yourself.
        """
        
        t0             = time.time()
        stim_id        = 0
        
        while True:
            t_diff = (self.stim_period - (time.time() - t0))/1000
            if t_diff > 0:
                time.sleep(t_diff)
            t0 += self.stim_period/1000
            
            missed_stimuli = self.missed_stimuli
            while response_queue.qsize() > 0:
                response_queue.get()
            if stimulus_queue.qsize() > 0:
                while stimulus_queue.qsize() > 0:
                    action = stimulus_queue.get()
                self.missed_stimuli = 0
            else:
                self.missed_stimuli += 1
                action = np.zeros((self.action_dim),dtype=int)
                
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
            
            stim_id += 1
            
            response_queue.put([response,
                                missed_stimuli,
                                spikes,
                                elecs,
                                stim_id])

    def reset(self,seed=None,options=None):
        """
        Reset the environment state.
        """
        
        super().reset(seed=seed)
        
        # Initialize the state by setting it to 0.
        self.state          = np.zeros((self.state_dim,))
        self.missed_stimuli = 0
        return self.state,{}

    def step(self,action):
        """
        Apply action and return new state, reward, termination info, and extra info. This process is not time sensitive (i.e. waits for user).
        """
        
        # Check action:
        action, msg = check_action(action,self.action_dim)
        
        # Apply action and get response 
        while self.response_queue.qsize() > 0:
            self.response_queue.get() # Queue needs to be emptied, if it has elements inside (happens when action is not sent in time)
        self.stimulus_queue.put(action)
        response,missed_stimuli,spikes,elecs,stim_id = self.response_queue.get()     
        
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
                "missed_cyc": missed_stimuli, 
                "stim_id": stim_id,
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
