import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import deque
import os
import math

# Parameters
STRATEGY_ID = 3


MAX_EPISODES = 5000
TARGET_UPDATE = 10 # Update frequency for the target network
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64 # Size of the experience replay batch
MEMORY_CAPACITY = 50000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_RATE = 0.9995
POLE_LENGTHS_SET = np.linspace(0.4, 1.8, 30)
NUM_PHASES_CURRICULUM = 4

if not os.path.exists('weights'):
    os.makedirs('weights')

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#Third strategy: Sorted replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = max_prio ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        N = len(self.buffer)
        weights = (N * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        state_tensors = torch.tensor(np.array(states), dtype=torch.float32)
        next_state_tensors = torch.tensor(np.array(next_states), dtype=torch.float32)

        return (state_tensors, 
                torch.tensor(actions).unsqueeze(1), 
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1), 
                next_state_tensors, 
                torch.tensor(dones, dtype=torch.uint8).unsqueeze(1), 
                indices, 
                torch.tensor(weights, dtype=torch.float32).unsqueeze(1))

    def update_priorities(self, batch_indices, td_errors, epsilon=1e-5):
        for idx, error in zip(batch_indices, td_errors):
            self.priorities[idx] = (abs(error) + epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        state_tensors = torch.tensor(np.array(states), dtype=torch.float32)
        next_state_tensors = torch.tensor(np.array(next_states), dtype=torch.float32)

        return (state_tensors, 
                torch.tensor(actions).unsqueeze(1), 
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1), 
                next_state_tensors, 
                torch.tensor(dones, dtype=torch.uint8).unsqueeze(1),
                None,
                None)

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, *args):
        pass

class DQNAgent:
    def __init__(self, env, strategy_id):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.strategy_id = strategy_id
        
        self.policy_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        if self.strategy_id == 3:
            self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY)
            self.beta = 0.4 
            self.beta_increment = (1.0 - self.beta) / MAX_EPISODES 
        else:
            self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        self.epsilon = EPS_START

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_dim)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(BATCH_SIZE)
        
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
        expected_state_action_values = (next_state_values * GAMMA * (1 - dones)) + rewards
        
        td_errors = torch.abs(state_action_values - expected_state_action_values).squeeze().tolist()
        
        if self.strategy_id == 3:
            loss = (is_weights * self.criterion(state_action_values, expected_state_action_values)).mean()
            self.memory.update_priorities(indices, td_errors)
        else:
            loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self, episode):
        if self.strategy_id in [1, 3]:
            self.epsilon = max(EPS_END, EPS_START * (EPS_DECAY_RATE ** episode))

        if self.strategy_id == 3:
            self.beta = min(1.0, self.beta + self.beta_increment)


def get_pole_length(strategy_id, episode):
    """
    Determines the pole length for the current episode based on the strategy.
    Strategies 2 and 3 now sample from the 30 discrete pole lengths.
    """
    
    if strategy_id == 1:
        # Strategy 1
        phase_size = len(POLE_LENGTHS_SET) // NUM_PHASES_CURRICULUM
        
        episode_per_phase = MAX_EPISODES // NUM_PHASES_CURRICULUM
        
        if episode < episode_per_phase:
            lengths_to_use = POLE_LENGTHS_SET[:phase_size]
        elif episode < 2 * episode_per_phase:
            lengths_to_use = POLE_LENGTHS_SET[phase_size:2*phase_size]
        elif episode < 3 * episode_per_phase:
            lengths_to_use = POLE_LENGTHS_SET[2*phase_size:3*phase_size]
        else:
            lengths_to_use = POLE_LENGTHS_SET[3*phase_size:]
            
        return random.choice(lengths_to_use)
            
    elif strategy_id in [2, 3]:
        return random.choice(POLE_LENGTHS_SET)
        
    else:
        return 0.5


def calculate_shaped_reward(reward, state):
    """
    Applies the custom reward for Strategy 3.
    """
    if STRATEGY_ID != 3:
        return reward
        
    pole_angle = state[2]
    pole_angular_vel = state[3]
    
    THETA_MAX = math.radians(12) 
    ALPHA = 1.5
    BETA = 0.05

    angle_bonus = ALPHA * (1.0 - abs(pole_angle) / THETA_MAX)
    angular_velocity_penalty = BETA * abs(pole_angular_vel)
    
    shaped_reward = reward + angle_bonus - angular_velocity_penalty
    
    return shaped_reward


def train_agent():
    env = gym.make('CartPole-v1') 
    
    agent = DQNAgent(env, STRATEGY_ID)
    
    save_path = f"weights/dql_strategy_{STRATEGY_ID}.pth"
    print(f"Strategy {STRATEGY_ID}")
    
    episode_rewards = deque(maxlen=100)
    
    for episode in range(1, MAX_EPISODES + 1):
        
        current_length = get_pole_length(STRATEGY_ID, episode)
        env.unwrapped.length = current_length
        
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            if STRATEGY_ID == 3:
                reward = calculate_shaped_reward(reward, next_state)
            
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if len(agent.memory) > BATCH_SIZE:
                agent.train_step()
                
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()
            
        agent.update_epsilon(episode)

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
            
        # Logging
        if episode % 100 == 0 or episode == MAX_EPISODES:
            print(f"Episode: {episode:4d}/{MAX_EPISODES} | "
                  f"Avg Reward (100): {avg_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Pole Length: {current_length:.2f}")

    print(f"Training finished")
    torch.save(agent.policy_net.state_dict(), save_path)
    env.close()

if __name__ == "__main__":
    train_agent()