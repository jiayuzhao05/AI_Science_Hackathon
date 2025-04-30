import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Define the experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience',
            ['state', 'action', 'reward', 'next_state', 'done'])

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_count = 0

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Initialize metrics
        self.losses = []
        self.rewards = []

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay buffer
        experiences = self.memory.sample(self.batch_size)
        batch = self.memory.Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store metrics
        self.losses.append(loss.item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

def train_agent(env, agent, n_episodes):
    """Train the agent."""
    episode_rewards = []

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition and update
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state

        episode_rewards.append(episode_reward)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return episode_rewards

if __name__ == "__main__":
    # Hyperparameters
    n_episodes = 1000
    learning_rate = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    buffer_size = 100000
    batch_size = 64
    target_update = 10

    # Create environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, './video')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update
    )

    # Train the agent
    rewards = train_agent(env, agent, n_episodes)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('dqn_training_rewards.png')
    plt.close()

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(agent.losses)
    plt.title('Training Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.savefig('dqn_training_losses.png')
    plt.close() (rl+bnpytorch) [jiayuzhao@midway3-login4 Madeleine_team]$