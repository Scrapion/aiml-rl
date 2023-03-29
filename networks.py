import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.distributions.categorical import Categorical


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 action_size):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims                                    # Osservazioni dell'ambiente
        self.fc1_dims = fc1_dims                                        # Neuroni del primo layer
        self.fc2_dims = fc2_dims                                        # Neuroni del secondo layer
        self.action_size = action_size                                  # Numero di azioni possibili
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)           # Primo Hidden Layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)              # Secondo Hidden Layer

        self.move = nn.Linear(self.fc2_dims, self.action_size[0])       # Output Layer per Move
        self.rotate = nn.Linear(self.fc2_dims, self.action_size[1])     # Output Layer per Rotate
        self.chase = nn.Linear(self.fc2_dims, self.action_size[2])      # Output Layer per Chase
        self.cast = nn.Linear(self.fc2_dims, self.action_size[3])       # Output Layer per Cast
        self.change = nn.Linear(self.fc2_dims, self.action_size[4])     # Output Layer per Change

        self.optimizer = optim.Adam(self.parameters(), lr=lr)           # Adam Optimizer
        self.loss = nn.MSELoss()                                        # Mean Square Error Loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = [self.move(x), self.rotate(x), self.chase(x), self.cast(x), self.change(x)]

        return actions

class PPOActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(PPOActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.head = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU())
        
        self.move = nn.Sequential(
                nn.Linear(fc2_dims, n_actions[0]),
                nn.Softmax(dim=-1))

        self.rotate = nn.Sequential(
                nn.Linear(fc2_dims, n_actions[1]),
                nn.Softmax(dim=-1))
                        
        self.chase = nn.Sequential(
                nn.Linear(fc2_dims, n_actions[2]),
                nn.Softmax(dim=-1))  
              
        self.cast = nn.Sequential(
                nn.Linear(fc2_dims, n_actions[3]),
                nn.Softmax(dim=-1))
        
        self.change = nn.Sequential(
                nn.Linear(fc2_dims, n_actions[4]),
                nn.Softmax(dim=-1))        


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mid_res = self.head(state)
        dist = [Categorical(self.move(mid_res)),
                Categorical(self.rotate(mid_res)),
                Categorical(self.chase(mid_res)),
                Categorical(self.cast(mid_res)),
                Categorical(self.change(mid_res)),]
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class PPOCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(PPOCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))