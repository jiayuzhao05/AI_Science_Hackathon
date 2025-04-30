import gymnasium as gym
from gymnasium import spaces

from Reward.LinearReward              import LinearReward      as Reward
from StateReduction.StaticStateSimple import StaticStateSimple as State

from HelperFunctions.check_action import check_action

import multiprocessing

import numpy as np
import time

import pickle
import zmq

from auth import Auth

class RealNetworkSync(gym.Env):
    """
    Create a time sensitive Simulated network environment, that can be interacted with just like the a real network. The sync case - different from the base 
    case - will give you your stimulation results in a periodic fashion (just like the biological networks). You can give this class up to 5 parameters:
    - action_dim       int               Number of stimuli used per applied stimulation pattern (i.e. action). Each stimulus is in {0,1,2,3,4}
    - state_dim        int               State dimension of state/observation space (reduced representation of neuronal activity)
    - circuit_id       int               Each group has 4 networks. Here, you choose which of the 4 networks you use. Must be in {0,1,2,3}
    - reward_function  func  (optional)  Reward function that receives the neuronal activity as an input. If you do not set it, the default is being used.
    - state_function   func  (optional)  This function transforms your neuronal activity into a reduced representation in [-1,1]^n, where [-1,1] are numbers
                                         between -1 and +1 and n is the state_dim defined above.
    
    You can interact with the environment as with any other Gymnasium environment: https://gymnasium.farama.org/index.html
    
    This is the environment that you should use to interact with the biological networks.
    
    If you do not send an action in time, the default action is being used, which does not stimulate the network at all (all action slots are zero).
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self,action_dim,state_dim,circuit_id,reward_object=None,state_object=None):
        super(RealNetworkSync,self).__init__()

        self.action_dim  =  action_dim
        self.state_dim   =   state_dim
        
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
        self.last_stimulus  = 0    

        # Server access (group_id and password are from auth.py and need to be adapted to your group)
        self.auth           = Auth()
        self.group_id       = self.auth.group_id
        self.username       = f"group_{self.group_id}"
        self.password       = self.auth.password
        self.host           = self.auth.host
        self.port           = self.auth.port
        circuit_id          = int(circuit_id+0.5)
        assert circuit_id  >= 0 # Your circuit id must be 0,1,2, or 3
        assert circuit_id  <= 3 # Your circuit id must be 0,1,2, or 3
        self.circuit_id     = circuit_id

        self.context                      = zmq.Context()
        self.dealer_socket                = self.context.socket(zmq.DEALER)
        self.dealer_socket.plain_username = self.username.encode()
        self.dealer_socket.plain_password = self.password.encode()
        self.dealer_socket.connect(f"tcp://{self.host}:{self.port}")

    def reset(self,seed=None,options=None):
        """
        Reset the environment state.
        """
        super().reset(seed=seed)
        
        # Initialize the state by setting it to 0.
        self.state          = np.zeros((self.state_dim,))
        self.last_stimulus  = -1
        return self.state,{}

    def step(self,action):
        """
        Apply action and return new state, reward, termination info, and extra info. This process is not time sensitive (i.e. waits for user).
        """

        # Check action:
        action, msg = check_action(action,self.action_dim)
        
        # Apply action
        msg_body = pickle.dumps({"username": self.username, "action": action, "circuit_id": self.circuit_id})
        self.dealer_socket.send(msg_body)

        # Wait for response
        poller = zmq.Poller()
        poller.register(self.dealer_socket, zmq.POLLIN)
        socks = dict(poller.poll(2000))
        if self.dealer_socket in socks:
            # If the socket is ready, receive the response
            response = pickle.loads(self.dealer_socket.recv())
        else:
            # Handle the timeout scenario
            print("Timeout occured, reconnect ... (If this persists, check authentication. Then ask your mentor, if problem occurs for more than 10 min.)")

            self.context                      = zmq.Context()
            self.dealer_socket                = self.context.socket(zmq.DEALER)
            self.dealer_socket.plain_username = self.username.encode()
            self.dealer_socket.plain_password = self.password.encode()
            self.dealer_socket.connect("tcp://127.0.0.1:5335")
            
            return(self.step(action))

        action   = response["action"]
        if msg == 'none':
            msg  = response["message"] # Only look at message, if the action passed intial test. Otherwise, tell user why it failed
        index    = response["index"]
        elecs    = response["elecs"]
        spikes   = response["spikes"]
        real     = response["real"]

        if self.last_stimulus >= 0:
            missed_stimuli = index - 1 - self.last_stimulus
        else:
            missed_stimuli = 0
        self.last_stimulus = index
        stim_id            = index
        if missed_stimuli < 0:
            # This means a cycle is done and a reset occurs
            missed_stimuli = 0
            if msg == 'none':
                msg = "Cycle is over. A new cycle (with a new network has been chosen). Reset your network"

        response           = np.stack([spikes,elecs],1)

        # Define the space
        self.state  = self.state_object.get_state(response)
        
        # Calculate reward
        self.reward = self.reward_object.reward(response)

        # No truncation/termination
        terminated = False
        truncated  = False
        
        # Extra information to get information for the user
        info = {"spikes":     spikes, 
                "elecs":      elecs, 
                "action":     action,
                "missed_cyc": missed_stimuli, 
                "stim_id":    stim_id, 
                "simulated":  not real,
                "comment":    msg}

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
