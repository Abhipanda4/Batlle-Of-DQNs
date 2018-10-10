import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = input_shape[0]
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.Conv2d(512, num_actions, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.network(x)
        return x

class DuellingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuellingDQN, self).__init__()
        self.num_inputs = input_shape[0]
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(self.num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
