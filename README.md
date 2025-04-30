# Notes on the PyTorch environment

When you run 
```
source setup_rl_and_biological_networks.sh
```
A conda envrionemnt is created that uses the pytorch packages listed in rl_and_biological_networks_env.yml

This environment enables teams to build and run their own ML models. We think that Deep Q-Networks (harder) and Multi-Armed Bandits (easier) are excellent choices for this task. However, there might be even better alternatives out there waiting to be discovered.

We advice using pytorch, but you can also set up your own environments. We refer you to the docs for th list of possible tech stacks:
 - [PyTorch](https://github.com/pytorch)
 - [Flax](https://github.com/google/flax)/[JAX](https://github.com/google/jax)
 - [TensorFlow](https://github.com/tensorflow)


# Main Challenge Information

Your goal is to write a reinforcement learning (RL) agent, which stimulates a biological neuronal network (BNN) made out of real neurons. The better the BNN spikes in a clock-wise fashion, the higher your reward will be (see Reward/LinearReward.py for exact definition of the used reward).

The BNN RL environment is implemented as a gymnasium gym: https://gymnasium.farama.org/index.html

We advise you to first familiarize yourself with RL using simpler environments (e.g. LunarLander-v3 from gymnasium) to create RL agents before you start working with the real neurons.

Once you are familiar with how RL works, you can start using the example scripts in the Examples directory. The first 4 examples are simulated local BNNs. These are only there to see if your code is running. Once it does, you can use the real networks (see script 5 for how) to do that.

you can reset your environment with `env.reset()`. You will stimulate it with `env.step(action)`, where `action` is a 5 element numpy integer vector:
```
state, reward, terminated, truncated, info = env.step(action)
```
In this response `state` is the current state of your network, `reward` is the reward you received for the action you applied. You can ignore `terminated` and `truncated`, as they are not needed for our BNNs. Finally, `info` contains important information for you, specifically:
- spikes: (numpy vector) Contains the spike times of the action potentials recorded in ms. Data is in [0,20]
- elecs:  (numpy vector) Contains the locations where the corresponding spike in spikes has been recorded. Data is in {0,1,2,3}
- action: (numpy vector) Is the action that has been applied to the network
- missed_cyc: (int) Contains how many actions have been missed by you since the last time you stimulated the network. This information is important to you to know whether you were too slow and need to make a choice of your action faster. Do expect to miss some cycles, though, as there can always be some unexpected connection delays, that sporadically can happen in the internet.
- stim_id: (int) A counter counting for the current episode how many actions have been already applied. Data is in [0,28800). Once this resets to 0, you will get a new network. Plan accordingly!
- simulated: (bool) True, if you have a simulated network. False, if you have a real network
- comment: (string) Give you feedback, if something goes wrong


## Extra Challenge Information

Look at the slides for extra information.

While it is not required, you can write your own state and reward functions. In fact, to perform well, we highly advise this. Look at the Reward and StateReduction directory for examples for that.

Good luck!

