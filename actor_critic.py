
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.set_default_tensor_type('torch.FloatTensor')

class AdvantageActorCritic(object):
    """
    If using this joint AAC method need to work a bit harder for theoretical justification.
    """
    def __init__(self, policy, value_fn, gamma, tau):
        self.policy = policy
        self.value_fn = value_fn
        self.gamma = gamma
        self.tau = tau
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizer_value = optim.Adam(self.value_fn.parameters(), lr=0.01)

    def update(self, batch):
        """
        Do policy update for algorithm
        """
        print ("Update AAC")
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        bonuses = batch['bonuses']
        terminals = batch['terminals']
        next_observations = batch['next_observations']

        obs = Variable(torch.from_numpy(observations).float())
        # state value estimates
        values = self.value_fn(obs)
        R = torch.zeros(1, 1)
        values = torch.cat((values, Variable(R)))

        probs = self.policy(obs)
        log_probs = F.log_softmax(probs)
        # print ("Actions: ", actions.shape, actions[0].shape)

        policy_loss = 0
        value_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = self.gamma * R + float(rewards[i]) * (1 - terminals[i])
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * self.gamma * self.tau + delta_t

            policy_loss = policy_loss - log_probs[i][actions[i][0]] * Variable(gae)

        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()

        # print ("pol loss: ", policy_loss)
        # print ("val loss: ", value_loss)
        policy_loss.backward()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)

        self.optimizer_policy.step()
        self.optimizer_value.step()
