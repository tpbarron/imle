import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.FloatTensor')


################################################################################
# Policies
################################################################################

class GaussianMLPPolicy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(GaussianMLPPolicy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def act(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        action_mean, _, action_std = self.forward(Variable(obs))
        action = torch.normal(action_mean, action_std)
        return action

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class DiscreteMLPPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(DiscreteMLPPolicy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        # self.affine2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, num_actions)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def act(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        action_logits = self.forward(Variable(obs))
        probs = F.softmax(action_logits)
        return probs.multinomial()

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        # x = F.tanh(self.affine2(x))
        action_logits = self.action_head(x)
        return action_logits


################################################################################
# Value functions
################################################################################

class MLPValue(nn.Module):

    def __init__(self, num_inputs):
        super(MLPValue, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        # self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def encode(self, x):
        return F.tanh(self.affine1(x))

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        # x = F.tanh(self.affine2(x))
        state_values = self.value_head(x)
        return state_values

################################################################################
# Shared parameter Actor Critic models
################################################################################

class DiscreteMLPActorCritic(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(DiscreteMLPActorCritic, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        value = self.value_head(x)
        actions = self.action_head(x)

        return actions, value
