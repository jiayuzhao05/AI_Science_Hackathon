import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim

class MultiArmedBandit:
    def __init__(self, n_arms=10, true_rewards=None, reward_std=1.0):
        """
        Initialize a multi-armed bandit environment.

        Args:
            n_arms: Number of arms (actions)
            true_rewards: True mean rewards for each arm (if None, random values are used)
            reward_std: Standard deviation of the reward distribution
        """
        self.n_arms = n_arms
        self.reward_std = reward_std

        # Set true rewards (if not provided, use random values)
        if true_rewards is None:
            self.true_rewards = np.random.normal(0, 1, n_arms)
        else:
            self.true_rewards = true_rewards

        # Track the best arm
        self.best_arm = np.argmax(self.true_rewards)
        self.best_reward = self.true_rewards[self.best_arm]

        print(f"True rewards: {self.true_rewards}")
        print(f"Best arm: {self.best_arm}, Best reward: {self.best_reward}")

    def pull(self, arm):
        """
        Pull an arm and get a reward.

        Args:
            arm: The arm to pull (index)

        Returns:
            The reward received
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Arm {arm} is out of range [0, {self.n_arms-1}]")

        # Generate reward from a normal distribution
        reward = np.random.normal(self.true_rewards[arm], self.reward_std)
        return reward

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize an epsilon-greedy agent.

        Args:
            n_arms: Number of arms
            epsilon: Probability of exploration
        """
        self.n_arms = n_arms
        self.epsilon = epsilon

        # Initialize estimates and counts
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)

    def select_action(self):
        """
        Select an action using epsilon-greedy strategy.

        Returns:
            The selected arm
        """
        if random.random() < self.epsilon:
            # Explore: choose a random arm
            return random.randrange(self.n_arms)
        else:
            # Exploit: choose the arm with the highest estimate
            return np.argmax(self.estimates)

    def update(self, arm, reward):
        """
        Update the agent's estimates after receiving a reward.

        Args:
            arm: The arm that was pulled
            reward: The reward received
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update estimate using incremental formula
        self.estimates[arm] = (1 - 1/n) * self.estimates[arm] + (1/n) * reward

class UCB1Agent:
    def __init__(self, n_arms, c=2.0):
        """
        Initialize a UCB1 agent.

        Args:
            n_arms: Number of arms
            c: Exploration parameter
        """
        self.n_arms = n_arms
        self.c = c

        # Initialize estimates and counts
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.total_pulls = 0

    def select_action(self):
        """
        Select an action using UCB1 strategy.

        Returns:
            The selected arm
        """
        self.total_pulls += 1

        # If any arm hasn't been pulled yet, pull it
        if 0 in self.counts:
            return np.where(self.counts == 0)[0][0]

        # Calculate UCB values
        ucb_values = self.estimates + self.c * np.sqrt(np.log(self.total_pulls) / self.counts)

        # Select the arm with the highest UCB value
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """
        Update the agent's estimates after receiving a reward.

        Args:
            arm: The arm that was pulled
            reward: The reward received
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update estimate using incremental formula
        self.estimates[arm] = (1 - 1/n) * self.estimates[arm] + (1/n) * reward

class ThompsonSamplingAgent:
    def __init__(self, n_arms, prior_a=1, prior_b=1):
        """
        Initialize a Thompson Sampling agent.

        Args:
            n_arms: Number of arms
            prior_a: Prior alpha parameter for Beta distribution
            prior_b: Prior beta parameter for Beta distribution
        """
        self.n_arms = n_arms
        self.prior_a = prior_a
        self.prior_b = prior_b

        # Initialize parameters for Beta distribution
        self.a = np.ones(n_arms) * prior_a
        self.b = np.ones(n_arms) * prior_b

    def select_action(self):
        """
        Select an action using Thompson Sampling.

        Returns:
            The selected arm
        """
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.a, self.b)

        # Select the arm with the highest sample
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update the agent's parameters after receiving a reward.

        Args:
            arm: The arm that was pulled
            reward: The reward received (assumed to be binary 0 or 1)
        """
        # For Thompson Sampling, we need binary rewards
        # We'll convert the continuous reward to binary by thresholding
        binary_reward = 1 if reward > 0 else 0

        # Update parameters
        self.a[arm] += binary_reward
        self.b[arm] += (1 - binary_reward)

class NeuralBanditAgent:
    def __init__(self, n_arms, input_size=1, hidden_size=64, learning_rate=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize a Neural Bandit agent.

        Args:
            n_arms: Number of arms
            input_size: Size of the input feature vector
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for the optimizer
            device: Device to use for computation
        """
        self.n_arms = n_arms
        self.device = device

        # Define the neural network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_arms)
        ).to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Initialize counts
        self.counts = np.zeros(n_arms)

    def select_action(self, state=None, epsilon=0.1):
        """
        Select an action using the neural network.

        Args:
            state: The current state (if None, a default state is used)
            epsilon: Probability of exploration

        Returns:
            The selected arm
        """
        if random.random() < epsilon:
            # Explore: choose a random arm
            return random.randrange(self.n_arms)

        # If no state is provided, use a default state
        if state is None:
            state = np.zeros(1)

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get Q-values from the network
        with torch.no_grad():
            q_values = self.network(state_tensor).squeeze().cpu().numpy()

        # Select the arm with the highest Q-value
        return np.argmax(q_values)

    def update(self, state, arm, reward):
        """
        Update the neural network after receiving a reward.

        Args:
            state: The current state
            arm: The arm that was pulled
            reward: The reward received
        """
        self.counts[arm] += 1

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get current Q-values
        q_values = self.network(state_tensor)

        # Create target Q-values
        target_q_values = q_values.clone()
        target_q_values[0, arm] = reward

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def run_experiment(bandit, agent, n_steps, agent_name):
    """
    Run an experiment with a bandit and an agent.

    Args:
        bandit: The bandit environment
        agent: The agent
        n_steps: Number of steps to run
        agent_name: Name of the agent (for plotting)

    Returns:
        Rewards and regrets
    """
    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)
    cumulative_rewards = np.zeros(n_steps)
    cumulative_regrets = np.zeros(n_steps)

    for t in tqdm(range(n_steps), desc=f"Running {agent_name}"):
        # Select action
        if isinstance(agent, NeuralBanditAgent):
            # For NeuralBanditAgent, we need to provide a state
            arm = agent.select_action(state=np.array([t/n_steps]))  # Use normalized time as state
        else:
            arm = agent.select_action()

        # Get reward
        reward = bandit.pull(arm)

        # Update agent
        if isinstance(agent, NeuralBanditAgent):
            # For NeuralBanditAgent, we need to provide a state
            agent.update(state=np.array([t/n_steps]), arm=arm, reward=reward)
        else:
            agent.update(arm, reward)

        # Record reward and regret
        rewards[t] = reward
        regret = bandit.best_reward - bandit.true_rewards[arm]
        regrets[t] = regret

        # Update cumulative values
        if t > 0:
            cumulative_rewards[t] = cumulative_rewards[t-1] + reward
            cumulative_regrets[t] = cumulative_regrets[t-1] + regret
        else:
            cumulative_rewards[t] = reward
            cumulative_regrets[t] = regret

    return rewards, regrets, cumulative_rewards, cumulative_regrets

def plot_results(results, n_steps, agents, bandit):
    """
    Plot the results of the experiments with enhanced visualization options.

    Args:
        results: Dictionary of results for each agent
        n_steps: Number of steps
        agents: Dictionary of agents
        bandit: The bandit environment
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))

    # Plot average rewards
    plt.subplot(3, 3, 1)
    for agent_name, (rewards, _, _, _) in results.items():
        plt.plot(np.arange(n_steps), rewards, label=agent_name)
    plt.title('Average Rewards')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # Plot cumulative rewards
    plt.subplot(3, 3, 2)
    for agent_name, (_, _, cumulative_rewards, _) in results.items():
        plt.plot(np.arange(n_steps), cumulative_rewards, label=agent_name)
    plt.title('Cumulative Rewards')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)

    # Plot average regrets
    plt.subplot(3, 3, 3)
    for agent_name, (_, regrets, _, _) in results.items():
        plt.plot(np.arange(n_steps), regrets, label=agent_name)
    plt.title('Average Regrets')
    plt.xlabel('Steps')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid(True)

    # Plot cumulative regrets
    plt.subplot(3, 3, 4)
    for agent_name, (_, _, _, cumulative_regrets) in results.items():
        plt.plot(np.arange(n_steps), cumulative_regrets, label=agent_name)
    plt.title('Cumulative Regrets')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)

    # Plot true rewards vs estimated rewards for each agent
    plt.subplot(3, 3, 5)
    x = np.arange(bandit.n_arms)
    width = 0.8 / len(agents)

    plt.bar(x, bandit.true_rewards, width=width, label='True Rewards', color='black', alpha=0.7)

    for i, (agent_name, agent) in enumerate(agents.items()):
        if hasattr(agent, 'estimates'):
            plt.bar(x + (i+1)*width, agent.estimates, width=width, label=f'{agent_name} Estimates')

    plt.title('True vs Estimated Rewards')
    plt.xlabel('Arm')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # Plot arm selection counts for each agent
    plt.subplot(3, 3, 6)
    for i, (agent_name, agent) in enumerate(agents.items()):
        if hasattr(agent, 'counts'):
            plt.bar(x + i*width, agent.counts, width=width, label=agent_name)

    plt.title('Arm Selection Counts')
    plt.xlabel('Arm')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # Plot reward distribution for each arm
    plt.subplot(3, 3, 7)
    for arm in range(bandit.n_arms):
        # Generate samples from the true reward distribution
        samples = np.random.normal(bandit.true_rewards[arm], bandit.reward_std, 1000)
        plt.hist(samples, bins=20, alpha=0.5, label=f'Arm {arm}')

    plt.title('Reward Distribution for Each Arm')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    # Plot learning curves (average reward over time)
    plt.subplot(3, 3, 8)
    window_size = 100
    for agent_name, (rewards, _, _, _) in results.items():
        # Calculate moving average
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(moving_avg)), moving_avg, label=agent_name)

    plt.title(f'Learning Curves (Moving Average, Window={window_size})')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)

    # Plot regret comparison
    plt.subplot(3, 3, 9)
    for agent_name, (_, _, _, cumulative_regrets) in results.items():
        plt.plot(np.arange(n_steps), cumulative_regrets, label=agent_name)

    plt.title('Regret Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('multi_arm_bandit_results.png', dpi=300)
    plt.close()

    # Create additional plots for detailed analysis

    # Plot 1: Arm selection over time
    plt.figure(figsize=(15, 10))
    for i, (agent_name, agent) in enumerate(agents.items()):
        if hasattr(agent, 'counts'):
            plt.subplot(2, 3, i+1)
            plt.bar(x, agent.counts, color='skyblue')
            plt.title(f'{agent_name} Arm Selection')
            plt.xlabel('Arm')
            plt.ylabel('Count')
            plt.grid(True)

    plt.tight_layout()
    plt.savefig('arm_selection_distribution.png', dpi=300)
    plt.close()

    # Plot 2: Estimated vs True rewards
    plt.figure(figsize=(15, 10))
    for i, (agent_name, agent) in enumerate(agents.items()):
        if hasattr(agent, 'estimates'):
            plt.subplot(2, 3, i+1)
            plt.bar(x - width/2, bandit.true_rewards, width=width, label='True', color='black', alpha=0.7)
            plt.bar(x + width/2, agent.estimates, width=width, label='Estimated', color='skyblue')
            plt.title(f'{agent_name} True vs Estimated Rewards')
            plt.xlabel('Arm')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)

    plt.tight_layout()
    plt.savefig('reward_estimates.png', dpi=300)
    plt.close()

    # Plot 3: Performance comparison
    plt.figure(figsize=(15, 5))

    # Average reward
    plt.subplot(1, 3, 1)
    avg_rewards = [np.mean(rewards) for _, (rewards, _, _, _) in results.items()]
    plt.bar(range(len(agents)), avg_rewards, color='skyblue')
    plt.xticks(range(len(agents)), list(agents.keys()), rotation=45, ha='right')
    plt.title('Average Reward')
    plt.ylabel('Reward')
    plt.grid(True)

    # Final cumulative reward
    plt.subplot(1, 3, 2)
    final_rewards = [cumulative_rewards[-1] for _, (_, _, cumulative_rewards, _) in results.items()]
    plt.bar(range(len(agents)), final_rewards, color='skyblue')
    plt.xticks(range(len(agents)), list(agents.keys()), rotation=45, ha='right')
    plt.title('Final Cumulative Reward')
    plt.ylabel('Reward')
    plt.grid(True)

    # Final cumulative regret
    plt.subplot(1, 3, 3)
    final_regrets = [cumulative_regrets[-1] for _, (_, _, _, cumulative_regrets) in results.items()]
    plt.bar(range(len(agents)), final_regrets, color='skyblue')
    plt.xticks(range(len(agents)), list(agents.keys()), rotation=45, ha='right')
    plt.title('Final Cumulative Regret')
    plt.ylabel('Regret')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Parameters
    n_arms = 10
    n_steps = 10000
    n_experiments = 5

    # Create bandit
    bandit = MultiArmedBandit(n_arms=n_arms)

    # Create agents
    agents = {
        'Epsilon-Greedy (ε=0.1)': EpsilonGreedyAgent(n_arms, epsilon=0.1),
        'Epsilon-Greedy (ε=0.01)': EpsilonGreedyAgent(n_arms, epsilon=0.01),
        'UCB1 (c=2)': UCB1Agent(n_arms, c=2.0),
        'Thompson Sampling': ThompsonSamplingAgent(n_arms),
        'Neural Bandit': NeuralBanditAgent(n_arms)
    }

    # Run experiments
    all_results = {}

    for agent_name, agent in agents.items():
        print(f"\nRunning experiment with {agent_name}...")

        # Run multiple experiments and average the results
        all_rewards = np.zeros((n_experiments, n_steps))
        all_regrets = np.zeros((n_experiments, n_steps))
        all_cumulative_rewards = np.zeros((n_experiments, n_steps))
        all_cumulative_regrets = np.zeros((n_experiments, n_steps))

        for i in range(n_experiments):
            # Create a new bandit for each experiment
            experiment_bandit = MultiArmedBandit(n_arms=n_arms)

            # Create a new agent of the same type
            if isinstance(agent, EpsilonGreedyAgent):
                experiment_agent = EpsilonGreedyAgent(n_arms, epsilon=agent.epsilon)
            elif isinstance(agent, UCB1Agent):
                experiment_agent = UCB1Agent(n_arms, c=agent.c)
            elif isinstance(agent, ThompsonSamplingAgent):
                experiment_agent = ThompsonSamplingAgent(n_arms)
            elif isinstance(agent, NeuralBanditAgent):
                experiment_agent = NeuralBanditAgent(n_arms)

            # Run the experiment
            rewards, regrets, cumulative_rewards, cumulative_regrets = run_experiment(
                experiment_bandit, experiment_agent, n_steps, f"{agent_name} (Experiment {i+1})"
            )

            # Store the results
            all_rewards[i] = rewards
            all_regrets[i] = regrets
            all_cumulative_rewards[i] = cumulative_rewards
            all_cumulative_regrets[i] = cumulative_regrets

        # Average the results
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_regrets = np.mean(all_regrets, axis=0)
        avg_cumulative_rewards = np.mean(all_cumulative_rewards, axis=0)
        avg_cumulative_regrets = np.mean(all_cumulative_regrets, axis=0)

        # Store the averaged results
        all_results[agent_name] = (avg_rewards, avg_regrets, avg_cumulative_rewards, avg_cumulative_regrets)

    # Plot the results with enhanced visualization
    plot_results(all_results, n_steps, agents, bandit)

    print("\nExperiments completed. Results saved to:")
    print("- multi_arm_bandit_results.png (main results)")
    print("- arm_selection_distribution.png (arm selection analysis)")
    print("- reward_estimates.png (true vs estimated rewards)")
    print("- performance_comparison.png (performance metrics)") (rl+bnpytorch) [jiayuzhao@midway3-login4 Madeleine_team]$