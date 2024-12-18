import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_size, hidden_dims, output_size ):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)