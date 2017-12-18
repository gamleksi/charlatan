import torch.nn.functional as F
import torch.nn as nn

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))
from pytorch_a2c_ppo_acktr.model import MLPPolicy

class ModifiedMLPPolicy(MLPPolicy):
    def __init__(self, num_inputs, action_space):
        super(ModifiedMLPPolicy, self).__init__(num_inputs, action_space)
        
        self.a_fc1 = nn.Linear(num_inputs, 128)
        self.a_fc2 = nn.Linear(128, 128)
        self.a_fc3 = nn.Linear(128, 64)

        self.v_fc1 = nn.Linear(num_inputs, 128)
        self.v_fc2 = nn.Linear(128, 128)
        self.v_fc3 = nn.Linear(128, 128)
        self.v_fc4 = nn.Linear(128, 1)

        self.train()
        self.reset_parameters()
    
    def forward(self, inputs):
        
        inputs.data = self.obs_filter(inputs.data)
        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        x = F.tanh(x) 

        x = self.v_fc4(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)
        
        x = self.a_fc3(x)
        x = F.tanh(x)

        return value, x