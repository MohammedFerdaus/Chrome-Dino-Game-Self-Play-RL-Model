import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
import cv2

class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        # CNN layers for processing 4 stacked frames (4, 80, 80)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        # After conv1: (80-8)/4 + 1 = 19
        # After conv2: (19-4)/2 + 1 = 8
        # After conv3: (8-3)/1 + 1 = 6
        # So final size is 64 * 6 * 6 = 2304
        
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, output_size)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.gamma = 0.99
        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        self.lr = 0.00025
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(3).to(self.device)
        self.target_model = DQN(3).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Frame stacking
        self.frame_stack = deque(maxlen=4)
        self.update_target_frequency = 1000
        self.step_count = 0
    
    def preprocess_frame(self, screen):
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to correct orientation
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 80x80
        frame = cv2.resize(frame, (80, 80))
        
        # Normalize to 0-1
        frame = frame / 255.0
        
        return frame
    
    def get_state(self, screen):
        frame = self.preprocess_frame(screen)
        
        # Initialize frame stack with first frame
        if len(self.frame_stack) == 0:
            for _ in range(4):
                self.frame_stack.append(frame)
        else:
            self.frame_stack.append(frame)
        
        # Stack 4 frames
        state = np.stack(self.frame_stack, axis=0)
        return state
    
    def reset_frame_stack(self):
        self.frame_stack.clear()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in mini_batch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in mini_batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in mini_batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in mini_batch])).to(self.device)
        dones = torch.BoolTensor(np.array([t[4] for t in mini_batch])).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename='model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games,
            'step_count': self.step_count
        }, filename)
    
    def load(self, filename='model.pth'):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0)
        self.n_games = checkpoint.get('n_games', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.model.eval()