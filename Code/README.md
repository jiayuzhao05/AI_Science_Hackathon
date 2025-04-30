# Authentification

Ask your mentor for your group id and password. Put it into auth.py

# Examples

5 examples of how to use the RL environments used in this project. Before you use these, learn how RL works using the [gymnasium](https://gymnasium.farama.org/index.html) environments for testing. Only use example 5 for the final step of the Hackathon.

# Gyms

Contains the code for the Gyms. You should not rewrite this code.

# HelperFunctions

Contains internally used helper functions for the project. You likely do not have to touch this part.

# Reward

Contains the reward function used in the final evaluation (LinearReward.py). You can write your own reward functions if you like. They need to be a class, that has a function that is called `reward(self,response)`. It will get a Nx2 vector, where the first row contains the spike times (spikes in info) and the second row the spike locations (elecs in info). It needs to return a scalar value (not necessarily integer).

# StateReduction

Classes that transform spiking activity into a compressed state. There are two approaches: (static = untrained) and (dynamic = trained). If you write your own state function, you need a class with a `def fit(self,...)` and a `def get_state(self,response)` function.

- fit: This one you can call to actually train your state function in the dynamic case. It can have as many arguments as you like
- get_state: Called every time there is a response from the BNN to a stimulus. The format of `response` is the same as described in the above Reward chapter
