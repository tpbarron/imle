import argparse
from collections import deque
import gym
import numpy as np
import torch
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
num_inputs = env.observation_space.shape[0]
# print (env.action_space)
num_actions = env.action_space.n #shape[0]

policy = models.MLPDiscretePolicy(num_inputs, num_actions)
value = models.MLPValue(num_inputs)

algo = actor_critic.AdvantageActorCritic(policy, value, args.gamma, args.tau)
memory = replay_memory.Memory(args.max_replay_size,
                              env.observation_space.shape,
                              1) #env.action_space.shape)
# print (num_inputs, num_actions)
dynamics = bnn.BNN(num_inputs+1, num_inputs, lr=args.bnn_lr, n_samples=args.bnn_n_samples)
kl_mean = deque(maxlen=args.kl_q_len)
kl_std = deque(maxlen=args.kl_q_len)
kl_previous = deque(maxlen=args.kl_q_len)

def compute_bnn_accuracy(inputss, targetss):
    acc = 0.
    for _inputs, _targets in zip(_inputss, _targetss):
        _out = dynamics.forward(Variable(torch.from_numpy(_inputs)).float())
        _out = _out.data.cpu().numpy()
        acc += np.mean(np.square(_out - _targets.reshape(_targets.shape[0], np.prod(_targets.shape[1:])) ))
    acc /= len(_inputss)
    return acc

def vime_bnn_update(inputss, targetss):
    print ("Old BNN accuracy: ", compute_bnn_accuracy(inputss, targetss))
    for inputs, targets in zip(_inputss, _targetss):
        dynamics.train(inputs, targets)
    print ("New BNN accuracy: ", compute_bnn_accuracy(inputss, targetss))

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

def imle_bnn_update(inputss, targetss):
    pass

def imle_bnn_bonus(inputs, targets):
    pass
    
for episode in range(args.n_iter):
    """
    Roll out batch and do RL update
    """
    print ("Running episode: ", episode)
    step = 0
    reward_batch = 0.
    num_episodes = 0
    # ensure that at least batch_size samples in memory after first epoch

    bonuses = []
    while step < args.batch_size:
        obs = env.reset()
        reward_episode = 0.
        for t in range(args.max_episode_steps):
            act = policy.act(obs).data.numpy()[0,0]
            next_obs, reward, done, info = env.step(act)
            memory.add_sample(obs, act, reward, done)
            reward_episode += reward

            # bonus = 0
            # if episode >= 0:
            #     # compute reward bonus
            #     if args.vime:
            #         bonus = vime_bnn_bonus(obs, act, next_obs)
            #     elif args.imle:
            #         bonus = imle_bnn_bonus(obs, act, next_obs)
            # bonuses.append(bonus)

            if args.render:
                env.render()

            if done:
                # bonuses.append(bonuses[-1]) # add last sample since obs[-1] has no next obs
                break

            obs = next_obs

        step += (t-1)
        num_episodes += 1
        reward_batch += reward_episode

    # bonuses = np.array(bonuses)
    # kl_previous.append(np.median(bonuses))
    # previous_mean_kl = np.mean(np.asarray(kl_previous))
    # for i in range(len(bonuses)):
    #     bonuses[i] = bonuses[i] / previous_mean_kl
    # print ("Bonuses: (mean) ", np.mean(bonuses), ", (std) ", np.std(bonuses))

    reward_batch /= num_episodes

    # # do bnn update if memory is large enough
    if memory.size >= args.min_replay_size:
        print ("Updating BNN")
        obs_mean, obs_std, act_mean, act_std = memory.mean_obs_act()
        _inputss, _targetss = [], []
        for _ in range(args.bnn_n_updates_per_sample):
            batch = memory.sample(args.bnn_batch_size)
            obs = (batch['observations'] - obs_mean) / (obs_std + 1e-8)
            next_obs = (batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
            act = (batch['actions'] - act_mean) / (act_std + 1e-8)

            _inputs = np.hstack([obs.reshape(obs.shape[0], -1), act])
            _targets = next_obs.reshape(next_obs.shape[0], -1)
            _inputss.append(_inputs)
            _targetss.append(_targets)

        # update bnn
        if args.vime:
            vime_bnn_update(_inputss, _targetss)
        elif args.imle:
            imle_bnn_update(_inputss, _targetss)

    # get last batch_size samples for update
    if memory.size > args.batch_size:
        batch = memory.sample_last(args.batch_size)
        algo.update(batch)

    if episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(episode, reward_episode, reward_batch))
