import argparse
from collections import deque
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.set_default_tensor_type('torch.FloatTensor')

import models
import replay_memory
import actor_critic
import bnn

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MountainCarContinuous-v1', help='Env to train.')
parser.add_argument('--batch-size', type=int, default=5000, help='RL pol grad batch')
parser.add_argument('--n-iter', type=int, default=250, help='Num iters')
parser.add_argument('--max-episode-steps', type=int, default=500, help='Max num ep steps')
parser.add_argument('--max-replay-size', type=int, default=100000, help='Max num samples to store in replay memory')
parser.add_argument('--render', action='store_true', help='Render env observations')
parser.add_argument('--vime', action='store_true', help='Do VIME update')
parser.add_argument('--imle', action='store_true', help='Do IMLE update')
parser.add_argument('--log-interval', type=int, default=1, help='Interval to print stats')
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor (default: 0.995)')
parser.add_argument('--tau', type=float, default=0.97, help='gae (default: 0.97)')
parser.add_argument('--bnn-n-updates-per-sample', type=int, default=500, help='num bnn updates per step')
parser.add_argument('--bnn-n-samples', type=int, default=10, help='num samples per bnn update')
parser.add_argument('--bnn-batch-size', type=int, default=10, help='batch size for bnn updates')
parser.add_argument('--bnn-lr', type=float, default=0.0001, help='lr for bnn updates')
parser.add_argument('--eta', type=float, default=0.0001, help='param balancing exploitation / explr')
parser.add_argument('--min-replay-size', type=int, default=500, help='Min replay size for update')
parser.add_argument('--kl-q-len', type=int, default=10, help='Past KL queue size used for normalization')
args = parser.parse_args()

assert not (args.vime and args.imle), "Cannot do both VIME and IMLE"

env = gym.make('CartPole-v1')

if args.vime:
    num_inputs = env.observation_space.shape[0]
    num_model_inputs = num_inputs
    num_actions = env.action_space.n #shape[0]
elif args.imle:
    num_inputs = env.observation_space.shape[0]
    num_model_inputs = 64
    num_actions = env.action_space.n #shape[0]


policy = models.DiscreteMLPPolicy(num_inputs, num_actions)
value_fn = models.MLPValue(num_inputs)

algo = actor_critic.AdvantageActorCritic(policy, value_fn, args.gamma, args.tau)
memory = replay_memory.Memory(args.max_replay_size,
                              env.observation_space.shape,
                              1) #env.action_space.shape)
# print (num_inputs, num_actions)
dynamics = bnn.BNN(num_model_inputs+1, num_model_inputs, lr=args.bnn_lr, n_samples=args.bnn_n_samples)
kl_mean = deque(maxlen=args.kl_q_len)
kl_std = deque(maxlen=args.kl_q_len)
kl_previous = deque(maxlen=args.kl_q_len)

def compute_bnn_accuracy(inputs, actions, targets):
    acc = 0.
    for inp, act, tar in zip(inputs, actions, targets):
        input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
        target_dat = tar.reshape(tar.shape[0], -1)

        _out = dynamics.forward(Variable(torch.from_numpy(input_dat)).float())
        _out = _out.data.cpu().numpy()
        acc += np.mean(np.square(_out - tar.reshape(tar.shape[0], np.prod(tar.shape[1:])) ))
    acc /= len(inputs)
    return acc

def vime_bnn_update(inputs, actions, targets):
    print ("Old BNN accuracy: ", compute_bnn_accuracy(inputs, actions, targets))
    for inp, act, tar in zip(inputs, actions, targets):
        input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
        target_dat = tar.reshape(tar.shape[0], -1)
        dynamics.train(input_dat, target_dat)
    print ("New BNN accuracy: ", compute_bnn_accuracy(inputs, actions, targets))

def vime_bnn_bonus(obs, act, next_obs):
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    obs = obs[np.newaxis,:]
    act = act[np.newaxis,:]
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    inputs = np.hstack([obs, act])
    targets = next_obs.reshape(next_obs.shape[0], -1)
    dynamics.save_old_params()
    bonus = dynamics.kl_given_sample(inputs, targets)
    bonus = bonus.data.cpu().numpy()
    dynamics.reset_to_old_params()
    return bonus

def imle_bnn_update(inputs, actions, targets):
    """ Main difference is that we first compute the feature representation
    given the states and then concat the action before training """
    print ("IMLE bnn update")
    # print ("Old BNN accuracy: ", compute_bnn_accuracy(inputs, actions, targets))
    for inp, act, tar in zip(inputs, actions, targets):
        inp_feat = value_fn.encode(Variable(torch.from_numpy(inp))).data.numpy()
        # print ("Inp feat: ", inp_feat.shape)
        tar_feat = value_fn.encode(Variable(torch.from_numpy(tar))).data.numpy()
        input_dat = np.hstack([inp_feat.reshape(inp_feat.shape[0], -1), act])
        # print ("inp dat:", input_dat.shape)
        target_dat = tar_feat.reshape(tar_feat.shape[0], -1)
        dynamics.train(input_dat, target_dat)
    # print ("New BNN accuracy: ", compute_bnn_accuracy(inputs, actions, targets))


def imle_bnn_bonus(obs, act, next_obs):
    """ Very similar to VIME. Look at infogain in feature space model """
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    obs = obs[np.newaxis,:]
    act = act[np.newaxis,:]

    obs_feat = value_fn.encode(Variable(torch.from_numpy(obs)).float()).data.numpy()
    next_obs_feat = value_fn.encode(Variable(torch.from_numpy(next_obs)).float()).data.numpy()

    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    inputs = np.hstack([obs_feat, act])
    targets = next_obs_feat.reshape(next_obs_feat.shape[0], -1)
    dynamics.save_old_params()
    bonus = dynamics.kl_given_sample(inputs, targets)
    bonus = bonus.data.cpu().numpy()
    dynamics.reset_to_old_params()
    return bonus


optimizer_policy = optim.Adam(policy.parameters(), lr=0.001)
optimizer_value = optim.Adam(value_fn.parameters(), lr=0.01)

obs = env.reset()
episode_reward = 0.
episode_step = 0
episode = 0

while True:
    step = 0

    values = []
    log_probs = []
    rewards = []
    entropies = []
    bonuses = []

    while step < args.batch_size:
        episode_step += 1
        obs_var = Variable(torch.from_numpy(obs).unsqueeze(0)).float()
        # print ("obs var: ", obs_var.size())
        logit = policy(obs_var)
        value = value_fn(obs_var)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        entropies.append(entropy)

        action = prob.multinomial().data
        # print ("Log prob: ", log_prob.size(), action.size())
        log_prob = log_prob.gather(1, Variable(action))

        action = action.numpy()
        next_obs, reward, done, info = env.step(action[0, 0])
        # print (obs, action, action.shape, reward, done)
        memory.add_sample(obs, action[0], reward, done)
        episode_reward += reward

        done = done or episode_step >= args.max_episode_steps

        if args.render:
            env.render()

        bonus = 0
        if episode >= 1:
            # compute reward bonus
            if args.vime:
                bonus = vime_bnn_bonus(obs, action[0], next_obs)
            elif args.imle:
                bonus = imle_bnn_bonus(obs, action[0], next_obs)
        bonuses.append(bonus)

        if done:
            print ("Episode reward: ", episode_reward)
            episode_reward = 0.
            episode_step = 0
            episode += 1
            next_obs = env.reset()
            bonuses.append(bonuses[-1])

        # print ("next_obs: ", obs.shape, next_obs.shape)
        obs = next_obs
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break

        step += 1

    # normalize bonuses
    bonuses = np.array(bonuses)
    kl_previous.append(np.median(bonuses))
    previous_mean_kl = np.mean(np.asarray(kl_previous))
    if previous_mean_kl > 0:
        for i in range(len(bonuses)):
            bonuses[i] = bonuses[i] / previous_mean_kl
    print ("Bonuses: (mean) ", np.mean(bonuses), ", (std) ", np.std(bonuses))

    rewards = [sum(x) for x in zip(rewards, np.squeeze(bonuses).tolist())]
    R = torch.zeros(1, 1)
    if not done:
        value = value_fn(Variable(torch.from_numpy(obs)).float().unsqueeze(0))
        R = value.data
    values.append(Variable(R))

    algo.update(values, log_probs, rewards, entropies)

    # do bnn update if memory is large enough
    if memory.size >= args.min_replay_size and episode % 5 == 0:
        print ("Updating BNN")
        obs_mean, obs_std, act_mean, act_std = memory.mean_obs_act()
        _inputss, _targetss, _actionss = [], [], []
        for _ in range(args.bnn_n_updates_per_sample):
            batch = memory.sample(args.bnn_batch_size)
            obs_data = (batch['observations'] - obs_mean) / (obs_std + 1e-8)
            next_obs_data = (batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
            act_data = (batch['actions'] - act_mean) / (act_std + 1e-8)

            _inputss.append(obs_data)
            _targetss.append(next_obs_data)
            _actionss.append(act_data)

        # update bnn
        if args.vime:
            vime_bnn_update(_inputss, _actionss, _targetss)
        elif args.imle:
            imle_bnn_update(_inputss, _actionss, _targetss)
