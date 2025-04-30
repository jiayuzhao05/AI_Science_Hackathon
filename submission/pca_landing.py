import gymnasium as gym
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt

class PCALanderAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        n_components: int = 2,  # Number of PCA components
        n_bins: int = 10,  # Number of bins for state discretization
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with PCA for state space reduction.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            n_components: Number of PCA components to keep
            n_bins: Number of bins for state discretization
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.n_components = n_components
        self.n_bins = n_bins

        # Initialize PCA and scaler
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

        # Initialize Q-values with reduced state space
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        # Store original states for PCA fitting
        self.state_buffer = []

        # Store bin edges for discretization
        self.bin_edges = None

        # Cache for transformed states
        self.state_cache = {}

    def fit_pca(self, n_episodes: int = 1000):
        """Collect states and fit PCA."""
        print("Collecting states for PCA fitting...")
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                self.state_buffer.append(obs)
                action = self.env.action_space.sample()
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        # Fit scaler and PCA
        self.state_buffer = np.array(self.state_buffer)
        self.scaler.fit(self.state_buffer)
        scaled_states = self.scaler.transform(self.state_buffer)
        self.pca.fit(scaled_states)

        # Get reduced states
        reduced_states = self.pca.transform(scaled_states)

        # Compute bin edges for each component
        self.bin_edges = []
        for i in range(self.n_components):
            edges = np.linspace(
                reduced_states[:, i].min() - 1e-6,
                reduced_states[:, i].max() + 1e-6,
                self.n_bins + 1
            )
            self.bin_edges.append(edges)

        # Print explained variance ratio
        print("Explained variance ratio:", self.pca.explained_variance_ratio_)
        print("Total explained variance:", np.sum(self.pca.explained_variance_ratio_))

        # Clear buffer after fitting
        self.state_buffer = []

    def transform_state(self, state):
        """Transform state using fitted PCA and discretize."""
        # Convert state to tuple for hashing
        state_tuple = tuple(state)

        # Check cache first
        if state_tuple in self.state_cache:
            return self.state_cache[state_tuple]

        # Transform state
        scaled_state = self.scaler.transform(state.reshape(1, -1))
        reduced_state = self.pca.transform(scaled_state)

        # Discretize
        discretized = []
        for i, value in enumerate(reduced_state.flatten()):
            bin_idx = np.digitize(value, self.bin_edges[i]) - 1
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))
            discretized.append(bin_idx)

        # Cache and return result
        result = tuple(discretized)
        self.state_cache[state_tuple] = result
        return result

    def get_action(self, obs: np.ndarray) -> int:
        """Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        reduced_state = self.transform_state(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[reduced_state]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Updates the Q-value of an action."""
        reduced_state = self.transform_state(obs)
        reduced_next_state = self.transform_state(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[reduced_next_state])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[reduced_state][action]
        )

        self.q_values[reduced_state][action] = (
            self.q_values[reduced_state][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def train_agent(env, agent, n_episodes):
    """Train the agent."""
    from tqdm import tqdm

    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

    return episode_rewards

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    n_components = 2  # Number of PCA components
    n_bins = 10  # Number of bins for state discretization

    # Create environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, './video')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # Create and train agent
    agent = PCALanderAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        n_components=n_components,
        n_bins=n_bins
    )

    # Fit PCA first
    agent.fit_pca(n_episodes=1000)

    # Train the agent
    rewards = train_agent(env, agent, n_episodes)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_rewards.png')
    plt.close()

    # Plot training error
    plt.figure(figsize=(10, 5))
    plt.plot(agent.training_error)
    plt.title('Training Error')
    plt.xlabel('Update Step')
    plt.ylabel('TD Error')
    plt.savefig('training_error.png')
(rl+bnpytorch) [jiayuzhao@midway3-login4 Madeleine_team]$