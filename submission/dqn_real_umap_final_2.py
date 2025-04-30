import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch.amp as amp
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
import umap
import warnings

import sys
sys.stdout.flush()

# Suppress specific scikit-learn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')
# Suppress UMAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='umap.umap_')

# Add parent directory to path for imports
current_dir = Path().resolve()
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Gyms.RealNetworkSync import RealNetworkSync
from Gyms.SimulatedNetworkSync import SimulatedNetworkSync
from StateReduction.DynamicStateUMAP import DynamicStateUMAP
from Reward.ExponentialReward import ExponentialReward
from Reward.ClockwiseStreakReward import ClockwiseStreakReward

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # Common feature layer with LayerNorm
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 34),
            nn.LayerNorm(34),  # Add LayerNorm after linear layer
            nn.ReLU()
        ).to(device)
        
        # Value stream with LayerNorm
        self.value_stream = nn.Sequential(
            nn.Linear(34, 34),
            nn.LayerNorm(34),  # Add LayerNorm after linear layer
            nn.ReLU(),
            nn.Linear(34, 1)
        ).to(device)
        
        # Advantage stream with LayerNorm
        self.advantage_stream = nn.Sequential(
            nn.Linear(34, 34),
            nn.LayerNorm(34),  # Add LayerNorm after linear layer
            nn.ReLU(),
            nn.Linear(34, action_dim)
        ).to(device)

    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams with LayerNorm
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, reward_bonus=1.0):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.reward_bonus = reward_bonus  # Bonus factor for non-zero rewards
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        # Track reward statistics
        self.zero_reward_count = 0
        self.nonzero_reward_count = 0
        # Track reward statistics for normalization with larger window
        self.reward_window_size = 1000  # Increased from implicit small window to 1000
        self.reward_window = deque(maxlen=self.reward_window_size)  # Use deque for efficient window management
        
        # Improved reward statistics tracking with exponential moving averages
        self.reward_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'std': 0.0,
            'count': 0,
            'ema_mean': 0.0,  # Exponential moving average of mean
            'ema_std': 1.0,   # Exponential moving average of std
            'ema_alpha': 0.01  # EMA decay factor (smaller = more stable)
        }
        
    def update_reward_stats(self, reward):
        """Update reward statistics using a sliding window and exponential moving averages"""
        # Add reward to window
        self.reward_window.append(reward)
        
        # Update min/max
        self.reward_stats['min'] = min(self.reward_stats['min'], reward)
        self.reward_stats['max'] = max(self.reward_stats['max'], reward)
        
        # Update count
        self.reward_stats['count'] += 1
        
        # Update exponential moving averages for more stable statistics
        alpha = self.reward_stats['ema_alpha']
        
        # Update EMA mean
        if self.reward_stats['count'] == 1:
            self.reward_stats['ema_mean'] = reward
        else:
            self.reward_stats['ema_mean'] = (1 - alpha) * self.reward_stats['ema_mean'] + alpha * reward
        
        # Update EMA std (using Welford's online algorithm)
        if self.reward_stats['count'] > 1:
            # Calculate squared difference from current mean
            diff = reward - self.reward_stats['ema_mean']
            # Update variance estimate
            self.reward_stats['ema_std'] = (1 - alpha) * self.reward_stats['ema_std'] + alpha * (diff * diff)
        
        # Only update full window statistics when window is full
        if len(self.reward_window) == self.reward_window_size:
            # Calculate statistics from window
            window_array = np.array(self.reward_window)
            self.reward_stats['mean'] = np.mean(window_array)
            self.reward_stats['std'] = np.std(window_array)
        
    def normalize_reward(self, reward):
        """Normalize reward using robust statistics with outlier handling"""
        # Use EMA statistics for normalization
        ema_mean = self.reward_stats['ema_mean']
        ema_std = max(np.sqrt(self.reward_stats['ema_std']), 1e-8)  # Ensure std is positive
        
        # Handle outliers by clipping extreme values before normalization
        if self.reward_stats['count'] > 10:  # Only apply after some initial data
            # Calculate z-score of current reward
            z_score = (reward - ema_mean) / ema_std
            
            # Clip extreme outliers before normalization
            if abs(z_score) > 5.0:  # Threshold for outlier detection
                # Clip to 5 standard deviations
                reward = ema_mean + 5.0 * ema_std * np.sign(z_score)
        
        # Normalize reward using EMA statistics
        normalized = (reward - ema_mean) / ema_std
        
        # Clip normalized reward to reasonable range
        return np.clip(normalized, -3, 3)
        
    def push(self, state, action, reward, next_state, done):
        # Update reward statistics
        self.update_reward_stats(reward)
        
        # Normalize reward
        normalized_reward = self.normalize_reward(reward)
        print(f"Normalized reward: {normalized_reward}")
        # Calculate priority based on normalized reward
        if normalized_reward < 0:
            # Give higher priority to less negative rewards
            priority = (1.0 / (abs(normalized_reward) + 1e-6)) ** self.alpha
            self.nonzero_reward_count += 1
        elif normalized_reward > 0:
            # Give highest priority to positive rewards
            priority = (normalized_reward + self.reward_bonus) ** (self.alpha * 1.5)
            self.nonzero_reward_count += 1
        else:
            # Give lowest priority to zero rewards
            priority = 1  # Fixed low priority for zero rewards
            self.zero_reward_count += 1
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, normalized_reward, next_state, done))
        else:
            # Replace oldest zero-reward experience if possible
            if normalized_reward != 0:
                # Find oldest zero-reward experience
                zero_indices = [i for i, x in enumerate(self.memory) if x[2] == 0]
                if zero_indices:
                    self.position = zero_indices[0]
                
            self.memory[self.position] = (state, action, normalized_reward, next_state, done)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
        # Print statistics periodically
        total = self.zero_reward_count + self.nonzero_reward_count
        if total % 1000 == 0:
            print(f"\nReplay Buffer Statistics:", flush=True)
            print(f"Zero rewards: {self.zero_reward_count} ({self.zero_reward_count/total*100:.1f}%)", flush=True)
            print(f"Non-zero rewards: {self.nonzero_reward_count} ({self.nonzero_reward_count/total*100:.1f}%)", flush=True)
            print(f"Reward stats - Min: {self.reward_stats['min']:.2f}, Max: {self.reward_stats['max']:.2f}, Mean: {self.reward_stats['mean']:.2f}", flush=True)
    
    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None, None
        
        # Calculate sampling probabilities with higher weight for non-zero rewards
        probs = self.priorities[:len(self.memory)]
        
        # Ensure non-zero rewards have higher sampling probability
        nonzero_indices = [i for i, x in enumerate(self.memory) if x[2] != 0]
        if nonzero_indices and len(nonzero_indices) < batch_size:
            # Force include all non-zero reward experiences
            remaining = batch_size - len(nonzero_indices)
            zero_indices = [i for i, x in enumerate(self.memory) if x[2] == 0]
            zero_probs = probs[zero_indices]
            zero_probs /= zero_probs.sum()
            additional_indices = np.random.choice(zero_indices, remaining, p=zero_probs)
            indices = np.concatenate([nonzero_indices, additional_indices])
        else:
            # Normal sampling with prioritization
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.memory[idx] for idx in indices]
        
        # Print batch statistics
        batch_rewards = [x[2] for x in batch]
        nonzero_count = sum(1 for r in batch_rewards if r != 0)
        print(f"\nBatch Statistics:", flush=True)
        print(f"Non-zero rewards in batch: {nonzero_count}/{batch_size} ({nonzero_count/batch_size*100:.1f}%)", flush=True)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Add bonus to priorities of non-zero reward experiences
            if self.memory[idx][2] != 0:  # if reward is non-zero
                priority = float(priority.item()) + self.reward_bonus
            else:
                priority = float(priority.item())
            self.priorities[idx] = priority
    
    def get_statistics(self):
        """Get current buffer statistics"""
        total = self.zero_reward_count + self.nonzero_reward_count
        return {
            'zero_reward_ratio': self.zero_reward_count / total if total > 0 else 0,
            'nonzero_reward_ratio': self.nonzero_reward_count / total if total > 0 else 0,
            'buffer_size': len(self.memory),
            'unique_experiences': len(set((tuple(x[0]), tuple(x[1])) for x in self.memory))
        }

class DQNAgent:
    def __init__(self, state_dim, action_dim, num_stimuli):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_stimuli = num_stimuli
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim * num_stimuli).to(device)
        self.target_net = DQN(state_dim, action_dim * num_stimuli).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-3  # Reduced initial learning rate
        self.target_update = 50  # More frequent target updates
        
        # Add Q-value clipping parameters for negative rewards
        self.max_q_value = 100.0  # Reduced maximum Q-value
        self.min_q_value = -100.0  # Reduced minimum Q-value
        self.td_error_clip = 10.0  # Reduced TD error clipping
        
        # Add reward shaping parameters
        self.reward_scale = 1.0  # Scale factor for rewards
        self.reward_shift = 0.0  # Shift factor for rewards
        self.reward_clip = (-10.0, 10.0)  # Clip rewards to this range
        
        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Prioritized Experience Replay with adjusted parameters
        self.memory = PrioritizedReplayBuffer(
            capacity=30000,
            reward_bonus=2.0,  # Reduced reward bonus
            alpha=0.6,  # Prioritization exponent
            beta=0.4,  # Importance sampling exponent
            beta_increment=0.001  # Beta increment per sample
        )
        
        # Initialize GPU optimization tools
        self.scaler = amp.GradScaler()
        
        # Add gradient accumulation steps
        self.grad_accumulation_steps = 4
        self.current_step = 0
        
        # Modified learning rate scheduler with more conservative settings
        self.total_training_steps = 28800
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=int(21600 / self.grad_accumulation_steps),
            pct_start=0.3,  # Longer warmup period
            cycle_momentum=False,
            div_factor=5.0,  # More conservative initial learning rate
            final_div_factor=10  # More conservative final learning rate
        )
        
        # Add gradient clipping parameters
        self.max_grad_norm = 1.0  # Maximum gradient norm for clipping
        
        # Track reward statistics
        self.reward_history = []
        self.avg_reward_window = 100
        
    def shape_reward(self, reward):
        """Apply reward shaping to help with learning from negative rewards"""
        # Scale and shift reward
        shaped_reward = reward * self.reward_scale + self.reward_shift
        
        # Clip reward to prevent extreme values
        shaped_reward = np.clip(shaped_reward, self.reward_clip[0], self.reward_clip[1])
        
        # Update reward statistics
        self.reward_history.append(shaped_reward)
        if len(self.reward_history) > self.avg_reward_window:
            self.reward_history.pop(0)
            
        # Adjust reward scale based on recent history with more conservative changes
        if len(self.reward_history) == self.avg_reward_window:
            avg_reward = np.mean(self.reward_history)
            if avg_reward < 0:
                # If average reward is negative, increase scale to make learning easier
                # More conservative increase
                self.reward_scale = min(2.0, self.reward_scale * 1.05)
            elif avg_reward > 0:
                # If average reward is positive, decrease scale to prevent explosion
                # More conservative decrease
                self.reward_scale = max(0.5, self.reward_scale * 0.95)
                
        return shaped_reward
        
    def remember(self, state, action, reward, next_state, done):
        # Apply reward shaping
        shaped_reward = self.shape_reward(reward)
        
        state = np.array(state).flatten()
        action = np.array(action).flatten()
        next_state = np.array(next_state).flatten()
        self.memory.push(state, action, shaped_reward, next_state, done)
        
    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim, size=self.num_stimuli)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            # Clip Q-values during action selection
            q_values = torch.clamp(q_values, self.min_q_value, self.max_q_value)
            q_values = q_values.view(-1, self.num_stimuli, self.action_dim)
            actions = q_values.argmax(dim=2)[0].cpu().numpy()
            return actions
    
    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return 0
        
        self.current_step += 1
        # Only zero gradients at start of accumulation
        if (self.current_step % self.grad_accumulation_steps) == 1:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Sample batch with priorities
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Stack all batch data at once using numpy - more efficient
        states = np.vstack([x[0] for x in batch])
        actions = np.vstack([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Convert to tensors in one go
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(device)
        
        # Use mixed precision training
        with amp.autocast(device_type='cuda', dtype=torch.float16):
            # Current Q values with clipping
            current_q_values = self.policy_net(states)
            current_q_values = torch.clamp(current_q_values, self.min_q_value, self.max_q_value)
            current_q_values = current_q_values.view(self.batch_size, self.num_stimuli, self.action_dim)
            current_q_values = current_q_values[torch.arange(self.batch_size).unsqueeze(1).to(device),
                                             torch.arange(self.num_stimuli).unsqueeze(0).to(device),
                                             actions.view(self.batch_size, self.num_stimuli)]
            current_q_values = current_q_values.sum(dim=1, keepdim=True)
            
            # Double DQN with clipping
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(next_states)
                next_actions = torch.clamp(next_actions, self.min_q_value, self.max_q_value)
                next_actions = next_actions.view(self.batch_size, self.num_stimuli, self.action_dim).argmax(dim=2)
                
                # Evaluate actions using target network
                next_q_values = self.target_net(next_states)
                next_q_values = torch.clamp(next_q_values, self.min_q_value, self.max_q_value)
                next_q_values = next_q_values.view(self.batch_size, self.num_stimuli, self.action_dim)
                next_q_values = next_q_values[torch.arange(self.batch_size).unsqueeze(1).to(device),
                                            torch.arange(self.num_stimuli).unsqueeze(0).to(device),
                                            next_actions]
                next_q_values = next_q_values.sum(dim=1, keepdim=True)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                target_q_values = torch.clamp(target_q_values, self.min_q_value, self.max_q_value)
            
            # Compute loss with importance sampling weights and TD error clipping
            td_errors = current_q_values - target_q_values
            td_errors = torch.clamp(td_errors, -self.td_error_clip, self.td_error_clip)
            loss = (weights * (td_errors ** 2)).mean()
            print(f"Loss: {loss.item()}")
        
        # Update priorities with clipped TD errors
        with torch.no_grad():
            new_priorities = (td_errors.abs() + 1e-6).cpu().numpy()
            self.memory.update_priorities(indices, new_priorities)
        
        # Scale loss by accumulation steps
        loss = loss / self.grad_accumulation_steps
        
        # Backward pass with scaled loss
        self.scaler.scale(loss).backward()
        
        # Only update weights after accumulating gradients
        if (self.current_step % self.grad_accumulation_steps) == 0:
            # First unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            
            # Then do optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Finally step the scheduler
            if self.current_step < 21599 and (int(self.current_step/self.grad_accumulation_steps) < int(21600 / self.grad_accumulation_steps)):
                self.scheduler.step()
            
            # Update target network more frequently
            if self.current_step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon more gradually
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item() * self.grad_accumulation_steps

    def get_statistics(self):
        """Get current agent statistics"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

def train_on_realsync_network():
    # Environment parameters
    state_dim = 4  # Changed from 16 to 4 to match UMAP
    action_dim = 5
    num_stimuli = 5
    circuit_id  = 0
    num_experiences = 1000
    # Create state reduction object with explicit n_components
    state_reduction = DynamicStateUMAP(state_dim=state_dim)
    reward_function = ExponentialReward()
    #reward_function = ClockwiseStreakReward()
    # Create environment with matching state dimension
    env = RealNetworkSync(action_dim=num_stimuli,
                         state_dim=state_dim,  \
                         circuit_id = circuit_id,
                         state_object=state_reduction,
                         reward_object=reward_function)
    
    # Create DQN agent
    agent = DQNAgent(state_dim=4,  # Matches UMAP dimension
                    action_dim=action_dim,
                    num_stimuli=num_stimuli)
    
    # Print all training parameters
    print("\n===== TRAINING PARAMETERS =====")
    print("\nEnvironment Parameters:")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Number of Stimuli: {num_stimuli}")
    print(f"Circuit ID: {circuit_id}")
    print(f"Number of Initial Experiences: {num_experiences}")
    
    print("\nAgent Parameters:")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Gamma (Discount Factor): {agent.gamma}")
    print(f"Initial Epsilon: {agent.epsilon}")
    print(f"Epsilon Minimum: {agent.epsilon_min}")
    print(f"Epsilon Decay: {agent.epsilon_decay}")
    print(f"Initial Learning Rate: {agent.learning_rate}")
    print(f"Target Network Update Frequency: {agent.target_update}")
    print(f"Max Q-Value: {agent.max_q_value}")
    print(f"Min Q-Value: {agent.min_q_value}")
    print(f"TD Error Clip: {agent.td_error_clip}")
    print(f"Reward Scale: {agent.reward_scale}")
    print(f"Reward Shift: {agent.reward_shift}")
    print(f"Reward Clip Range: {agent.reward_clip}")
    print(f"Gradient Accumulation Steps: {agent.grad_accumulation_steps}")
    print(f"Max Gradient Norm: {agent.max_grad_norm}")
    
    print("\nTraining Parameters:")
    print(f"Total Training Steps: {agent.total_training_steps}")
    #print(f"Number of Episodes: {episodes}")
    #print(f"Results Directory: {results_dir}")
    
    print("\nState Reduction Parameters:")
    print(f"UMAP State Dimension: {state_reduction.state_dim}")
    #print(f"UMAP n_neighbors: {state_reduction.n_neighbors}")
    #print(f"UMAP min_dist: {state_reduction.min_dist}")
    
    print("\n===== STARTING TRAINING =====\n")
    
    print("Collecting initial experiences for state reduction training...")
    spikes = []
    elecs = []
    
    # Collect initial experiences with progress bar
    with tqdm(total=num_experiences, desc="Collecting experiences") as pbar:
        while len(spikes) < num_experiences:  # Continue until we have exactly 1000 valid samples
            try:
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                # Print info for every sample
                print(f"Info: {info}", flush=True)
                print(f"State: {state}, Reward: {reward}")
                # # Skip empty spike arrays
                # if len(info['spikes']) == 0:
                #     continue
                    
                spikes.append(info['spikes'])
                elecs.append(info['elecs'])
                pbar.update(1)
            except Exception as e:
                print(f"Error during experience collection: {e}")
                continue
    
    print(f"Collected {len(spikes)} valid samples (non-empty spike arrays)")
    
    # Train the state reduction directly on the state_reduction object
    print("Training state reduction...")
    # Calculate appropriate n_neighbors based on dataset size
    n_neighbors = min(15, len(spikes) - 1) if len(spikes) > 1 else 1
    print(f"Using n_neighbors={n_neighbors} for UMAP")
    
    X_reduced = state_reduction.train(spikes, elecs, n_neighbors=n_neighbors, min_dist=0.1)
    
    # Create directory for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'umap_real_network_training_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot reduced data if 2D
    if state_dim >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('UMAP 2D Visualization of Neural States')
        plt.savefig(os.path.join(results_dir, "umap_visualization.png"))
        plt.close()
    
    # Training parameters
    episodes = 1000
    best_reward_per_action = float('-inf')  # Changed from best_reward
    rewards_history = []
    losses_history = []
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    
    # Training statistics
    stats = {
        'episode': [],
        'reward': [],
        'avg_reward': [],
        'rewards_per_action': [],
        'loss': [],
        'epsilon': [],
        'learning_rate': [],
        'memory_size': [],
        'episode_length': [],
        'network_id': []
    }
    
    print("\nStarting DQN training on real network...")
    progress_bar = tqdm(range(episodes), desc="Training Episodes")
    
    # Track the current network ID
    current_network_id = None
    
    state, _ = env.reset()
    state = np.array(state).flatten()

    for episode in progress_bar:
        try:
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.act(state)
                
                # Take action in environment
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = np.array(next_state).flatten()
                    done = terminated or truncated
                    
                    # Print info for every sample
                    print(f"Episode {episode} - Step {episode_steps} - Info: {info}", flush=True)
                    
                    # Check for empty actions or missed cycles
                    if 'action' in info and (info['action'] is None or len(info['action']) == 0):
                        print(f"WARNING: Empty action detected at step {episode_steps}", flush=True)
                    
                    if 'missed_cyc' in info and info['missed_cyc'] > 0:
                        print(f"WARNING: Missed cycles ({info['missed_cyc']}) at step {episode_steps}", flush=True)
                    
                    # Print state and reward information
                    print(f"State shape: {next_state.shape}, Reward: {reward}", flush=True)
                    
                    # Check if we've switched to a new network
                    stim_id = info.get('stim_id', None)
                    if stim_id is not None:
                        # If stim_id is 0 and we've already been training on a network, or if there's a comment
                        if (stim_id == 0 and current_network_id is not None) or (info.get('comment', '') == 'Cycle is over. A new cycle (with a new network has been chosen). Reset your network') or len(info['comment']) >= 20:
                        #if (stim_id == 0 and current_network_id is not None) or (info.get('comment', '') != 'none'):
                            print(f"\nDetected network switch or comment (stim_id: {stim_id}, comment: {info.get('comment', '')}). Retraining state reducer...")
                            
                            # Collect new data for retraining
                            new_spikes = []
                            new_elecs = []
                            # Collect initial experiences with progress bar
                            with tqdm(total=num_experiences, desc="Collecting new experiences") as pbar:
                                while len(new_spikes) < num_experiences:  # Continue until we have exactly 1000 valid samples
                                    try:
                                        action = env.action_space.sample()
                                        
                                        state, reward, terminated, truncated, info = env.step(action)
                                        # Print info for every sample
                                        print(f"Info: {info}", flush=True)
                                        print(f"State: {state}, Reward: {reward}")
                                        # # Skip empty spike arrays
                                        # if len(info['spikes']) == 0:
                                        #     continue
                                            
                                        new_spikes.append(info['spikes'])
                                        new_elecs.append(info['elecs'])
                                        pbar.update(1)
                                    except Exception as e:
                                        print(f"Error during experience collection: {e}")
                                        continue
                            
                            print(f"Collected {len(new_spikes)} valid samples (non-empty spike arrays)")
                            
                            # Retrain state reducer
                            # Calculate appropriate n_neighbors based on dataset size
                            n_neighbors = min(15, len(new_spikes) - 1) if len(new_spikes) > 1 else 1
                            print(f"Using n_neighbors={n_neighbors} for UMAP retraining")
                            
                            X_reduced = state_reduction.train(new_spikes, new_elecs, n_neighbors=n_neighbors, min_dist=0.1)
                            print(f"Retrained reduced data shape: {X_reduced.shape}")
                            
                            #state, _ = env.reset()
                            state = np.array(state).flatten()

                            # Reinitialize the agent
                            agent = DQNAgent(state_dim=4, action_dim=action_dim, num_stimuli=num_stimuli)
                            # Not needed to remember the state as we are not training the agent on the new network yet
                            #agent.remember(state, action, reward, next_state, done)
                            print("Agent reinitialized for new network")
                            
                            # Update current network ID
                            current_network_id = stim_id
                            done = True
                        else:
                            current_network_id = stim_id
                            
                            # Check if we've reached stim_id 21600
                            if stim_id >= 21599:
                                print("\nReached stim_id 21599. Stopping training and switching to inference mode.")
                                # Save the current model as the best model if it's better than previous best
                                if episode_steps > 1000:
                                    current_reward_per_action = episode_reward / episode_steps if episode_steps > 0 else float('-inf')
                                    if current_reward_per_action > best_reward_per_action:
                                        best_reward_per_action = current_reward_per_action
                                        torch.save(agent.policy_net.state_dict(), best_model_path)
                                        print('episode', episode,f"New best model saved with reward per action: {best_reward_per_action:.4f}")
                                    
                                # Load the best model for inference
                                if os.path.exists(best_model_path):
                                    print("Loading best model for inference...")
                                    try:
                                        agent.policy_net.load_state_dict(torch.load(best_model_path))
                                        print("Best model loaded successfully")
                                        agent.epsilon = 0.0  # Set epsilon to 0 for pure inference
                                    except Exception as e:
                                        print(f"Error loading best model: {e}")
                                        break
                                
                                # Continue with inference for the last 7200 samples
                                print(f"Starting inference for samples {stim_id} to {28800}")
                                # Close the training progress bar before starting inference
                                progress_bar.close()
                                
                                # Create a new progress bar for inference
                                inference_progress = tqdm(total=7200, desc="Inference Progress")
                                while stim_id < 28799:  # 21600 + 7200 = 28800
                                    try:
                                        action = agent.act(state)  # Use best model for inference
                                        next_state, reward, terminated, truncated, info = env.step(action)
                                        next_state = np.array(next_state).flatten()
                                        state = next_state
                                        stim_id = info.get('stim_id', stim_id)
                                        episode_reward += reward
                                        episode_steps += 1
                                        print(f"Info: {info}", flush=True)
                                        print(f"State: {state}, Reward: {reward}")
                                        # Update inference progress bar
                                        inference_progress.update(1)
                                        inference_progress.set_postfix({
                                            'reward': f'{episode_reward:.2f}',
                                            'steps': f'{episode_steps}',
                                            'stim_id': f'{stim_id}',
                                            'reward_per_action': f'{episode_reward / episode_steps:.2f}'
                                        })
                                        
                                        if terminated or truncated:
                                            print(f"Inference terminated: terminated={terminated}, truncated={truncated}")
                                            break

                                        if stim_id == 28799:
                                            print(f"Inference terminated at stim_id={stim_id}")
                                            print(f"Reward: {episode_reward}, Steps: {episode_steps}, Epsilon: {agent.epsilon}, Learning Rate: {agent.optimizer.param_groups[0]['lr']}, Memory Size: {len(agent.memory.memory)}, Network ID: {current_network_id}, Reward per action: {episode_reward / episode_steps}")
                                            break
                                    except Exception as e:
                                        print(f"Error during inference step: {e}")
                                        break
                                
                                # Close inference progress bar
                                inference_progress.close()
                                done = True
                                break
                            else:
                                # Store experience and train
                                agent.remember(state, action, reward, next_state, done)
                                loss = agent.train()
                        
                                if loss is not None:
                                    episode_loss += loss
                            
                                # Only increment steps and update state when we have valid spike data
                                episode_reward += reward
                                episode_steps += 1
                                state = next_state
                            
                                
                                done = (episode_steps % 1440 == 0)
                        
                except Exception as e:
                    print(f"Error during step: {e}")
                    done = True
                    continue
            
            # Calculate statistics
            avg_reward = np.mean(rewards_history[-10:] + [episode_reward]) if rewards_history else episode_reward
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            rewards_per_action = episode_reward / episode_steps if episode_steps > 0 else 0
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'rewards_per_action': f'{rewards_per_action:.2f}',
                'epsilon': f'{agent.epsilon:.3f}',
                'lr': f'{agent.optimizer.param_groups[0]["lr"]:.2e}',
                'network_id': f'{current_network_id}'
            })
            
            # Record statistics
            rewards_history.append(episode_reward)
            losses_history.append(avg_loss)
            
            stats['episode'].append(episode)
            stats['reward'].append(episode_reward)
            stats['avg_reward'].append(avg_reward)
            stats['rewards_per_action'].append(rewards_per_action)
            stats['loss'].append(avg_loss)
            stats['epsilon'].append(agent.epsilon)
            stats['learning_rate'].append(agent.optimizer.param_groups[0]['lr'])
            stats['memory_size'].append(len(agent.memory.memory))
            stats['episode_length'].append(episode_steps)
            stats['network_id'].append(current_network_id)

            # Save the model if it's the best so far
            if episode_steps > 1000:
                current_reward_per_action = episode_reward / episode_steps if episode_steps > 0 else float('-inf')
                if current_reward_per_action > best_reward_per_action:
                    best_reward_per_action = current_reward_per_action
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    print('episode', episode, f"New best model saved with reward per action: {best_reward_per_action:.4f}")

            # Save intermediate results
            if (episode + 1) % 5 == 0:
                df = pd.DataFrame(stats)
                df.to_csv(os.path.join(results_dir, 'training_stats_intermediate.csv'), index=False)
                create_training_visualization(df, results_dir, is_final=False)
                
        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            continue
    
    # Create final visualizations
    print("\nGenerating final visualizations...")
    df = pd.DataFrame(stats)
    create_training_visualization(df, results_dir, is_final=True)
    df.to_csv(os.path.join(results_dir, 'training_stats.csv'), index=False)
    
    # Print final statistics
    print("\nTraining Complete!")
    print(f"Results saved in: {results_dir}")
    print(f"Best reward per action: {best_reward_per_action:.4f}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Final learning rate: {agent.optimizer.param_groups[0]['lr']:.2e}")
    print(f"Final rewards per action: {rewards_per_action:.2f}")
    
    return rewards_history, results_dir

def create_training_visualization(df, results_dir, is_final=True):
    """Create comprehensive training visualizations"""
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    
    # Handle NaN values in the dataframe
    df_clean = df.copy()
    # Replace NaN values with 0 for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # 1. Training Progress Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DQN Training Progress Overview', fontsize=16)
    
    # Rewards plot
    sns.lineplot(data=df_clean, x='episode', y='reward', alpha=0.3, color='blue', ax=axes[0,0])
    sns.lineplot(data=df_clean, x='episode', y='avg_reward', color='green', ax=axes[0,0])
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].legend(['Episode Reward', '10-Episode Average'])
    
    # Rewards per action plot
    sns.lineplot(data=df_clean, x='episode', y='rewards_per_action', color='red', ax=axes[0,1])
    axes[0,1].set_title('Rewards per Action')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Rewards per Action')
    
    # Loss plot
    sns.lineplot(data=df_clean, x='episode', y='loss', color='green', ax=axes[1,0])
    axes[1,0].set_title('Training Loss')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Loss')
    
    # Epsilon plot
    sns.lineplot(data=df_clean, x='episode', y='epsilon', color='purple', ax=axes[1,1])
    axes[1,1].set_title('Epsilon (Exploration Rate)')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_overview.png'))
    plt.close()
    
    # 2. Learning Rate and Memory Size
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning rate
    sns.lineplot(data=df_clean, x='episode', y='learning_rate', color='blue', ax=axes[0])
    axes[0].set_title('Learning Rate Schedule')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_yscale('log')
    
    # Memory size
    sns.lineplot(data=df_clean, x='episode', y='memory_size', color='green', ax=axes[1])
    axes[1].set_title('Replay Buffer Size')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Memory Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_rate_memory.png'))
    plt.close()
    
    # 3. Reward Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_clean, x='reward', bins=30)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.savefig(os.path.join(results_dir, 'reward_distribution.png'))
    plt.close()
    
    # 4. Correlation Heatmap - Handle NaN values in correlation calculation
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_df = df_clean[['reward', 'avg_reward', 'loss', 'epsilon', 'episode_length', 'learning_rate', 'memory_size']]
    # Calculate correlation with NaN handling
    correlation = numeric_df.corr(method='pearson', min_periods=1)
    # Replace any remaining NaN values with 0
    correlation = correlation.fillna(0)
    
    # Create heatmap with explicit vmin and vmax to avoid warnings
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 5. Cumulative Rewards
    plt.figure(figsize=(10, 6))
    df_clean['cumulative_reward'] = df_clean['reward'].cumsum()
    sns.lineplot(data=df_clean, x='episode', y='cumulative_reward', color='blue')
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.savefig(os.path.join(results_dir, 'cumulative_rewards.png'))
    plt.close()
    
    # 6. Steps per Episode
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_clean, x='episode', y='episode_length', color='purple')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig(os.path.join(results_dir, 'steps_per_episode.png'))
    plt.close()
    
    # 7. Reward Per Action Over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_clean, x='episode', y='rewards_per_action', color='red')
    plt.title('Rewards per Action Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Rewards per Action')
    plt.savefig(os.path.join(results_dir, 'rewards_per_action.png'))
    plt.close()
    
    # 8. Network ID visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_clean, x='episode', y='network_id', color='orange')
    plt.title('Network ID Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Network ID')
    plt.savefig(os.path.join(results_dir, 'network_id.png'))
    plt.close()
    
    if is_final:
        # 9. Final Performance Analysis
        plt.figure(figsize=(12, 8))
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rewards over time
        sns.lineplot(data=df_clean, x='episode', y='reward', alpha=0.3, color='blue', ax=axes[0,0])
        sns.lineplot(data=df_clean, x='episode', y='avg_reward', color='green', ax=axes[0,0])
        axes[0,0].set_title('Rewards vs. Episode')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].legend(['Episode Reward', '10-Episode Average'])
        
        # Reward per action over time
        sns.lineplot(data=df_clean, x='episode', y='rewards_per_action', color='red', ax=axes[0,1])
        axes[0,1].set_title('Average Reward Per Action vs. Episode')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Reward Per Action')
        
        # Epsilon over time
        sns.lineplot(data=df_clean, x='episode', y='epsilon', color='purple', ax=axes[1,0])
        axes[1,0].set_title('Epsilon vs. Episode')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Epsilon')
        
        # Learning rate over time
        sns.lineplot(data=df_clean, x='episode', y='learning_rate', color='orange', ax=axes[1,1])
        axes[1,1].set_title('Learning Rate vs. Episode')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_vs_episode.png'))
        plt.close()

def analyze_training_results(results_dir):
    """Analyze training results and provide insights"""
    # Load training statistics
    stats_path = os.path.join(results_dir, 'training_stats.csv')
    if not os.path.exists(stats_path):
        print(f"Training statistics file not found: {stats_path}")
        return
    
    df = pd.read_csv(stats_path)
    
    # Calculate key metrics
    final_reward = df['reward'].iloc[-1]
    best_reward = df['reward'].max()
    avg_reward = df['reward'].mean()
    final_avg_reward = df['avg_reward'].iloc[-1]
    
    # Calculate total steps
    total_steps = df['episode_length'].sum()
    total_episodes = len(df)
    
    # Calculate learning progress
    early_rewards = df['reward'].iloc[:5].mean()
    late_rewards = df['reward'].iloc[-5:].mean()
    reward_improvement = (late_rewards - early_rewards) / (abs(early_rewards) + 1e-6) * 100
    
    # Print analysis
    print("\n===== TRAINING ANALYSIS =====")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"Best Episode Reward: {best_reward:.2f}")
    print(f"Final Episode Reward: {final_reward:.2f}")
    print(f"Average Episode Reward: {avg_reward:.2f}")
    print(f"Final 10-Episode Average Reward: {final_avg_reward:.2f}")
    print(f"Reward Improvement: {reward_improvement:.2f}%")
    
    # Check for convergence
    if reward_improvement > 0:
        print("✓ Model shows positive learning progress")
    else:
        print("✗ Model does not show clear learning progress")
    
    # Check for stability
    reward_std = df['reward'].std()
    if reward_std < avg_reward * 0.5:
        print("✓ Rewards are relatively stable")
    else:
        print("✗ Rewards show high variability")
    
    # Check for exploration
    final_epsilon = df['epsilon'].iloc[-1]
    if final_epsilon > 0.1:
        print("✓ Model still maintains good exploration")
    else:
        print("✗ Model has low exploration (epsilon)")
    
    # Recommendations
    print("\n===== RECOMMENDATIONS =====")
    if reward_improvement < 10:
        print("- Consider increasing the learning rate or extending training")
    if reward_std > avg_reward * 0.5:
        print("- Consider reducing the learning rate or increasing batch size")
    if final_epsilon < 0.1:
        print("- Consider increasing epsilon_min or epsilon_decay rate")
    
    # Save analysis to file
    analysis_path = os.path.join(results_dir, 'training_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write("===== TRAINING ANALYSIS =====\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Total Steps: {total_steps}\n")
        f.write(f"Best Episode Reward: {best_reward:.2f}\n")
        f.write(f"Final Episode Reward: {final_reward:.2f}\n")
        f.write(f"Average Episode Reward: {avg_reward:.2f}\n")
        f.write(f"Final 10-Episode Average Reward: {final_avg_reward:.2f}\n")
        f.write(f"Reward Improvement: {reward_improvement:.2f}%\n\n")
        
        f.write("===== RECOMMENDATIONS =====\n")
        if reward_improvement < 10:
            f.write("- Consider increasing the learning rate or extending training\n")
        if reward_std > avg_reward * 0.5:
            f.write("- Consider reducing the learning rate or increasing batch size\n")
        if final_epsilon < 0.1:
            f.write("- Consider increasing epsilon_min or epsilon_decay rate\n")
    
    print(f"\nAnalysis saved to: {analysis_path}")

if __name__ == "__main__":
    try:
        # Train on real network
        rewards, results_dir = train_on_realsync_network()
        
        # Analyze results
        analyze_training_results(results_dir)
        
        # Clean up any remaining resources
        plt.close('all')  # Close all matplotlib figures
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Force exit
        import sys
        sys.exit(0)

