# Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/enjoy.py

import argparse
import os
import gym
import pybullet_envs
import gym_x
import torch
from torch.autograd import Variable
import numpy as np
# import model
#
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10000000,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-path', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--log-dir', default='/tmp/gym/',
                    help='directory to save agent logs (default: /tmp/gym)')
args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

env = gym.make(args.env_name) #make_env(args.env_name, args.seed, 0, args.log_dir)()
env.seed(args.seed)

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_state = torch.zeros(1, *obs_shape)

actor_critic = torch.load(args.load_path)
print (actor_critic)
actor_critic.eval()
actor_critic = actor_critic.cpu()
if hasattr(actor_critic, 'obs_filter'):
    actor_critic.obs_filter.cpu()

def update_current_state(state):
    shape_dim0 = env.observation_space.shape[0]
    state = torch.from_numpy(state).float()
    if args.num_stack > 1:
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
    current_state[:, -shape_dim0:] = state

env.render('human')
state = env.reset()
update_current_state(state)
reward = 0.0

while True:
    value, action = actor_critic.act(Variable(current_state, volatile=True), deterministic=True)
    cpu_actions = action.data.cpu().numpy()
    if isinstance(env.action_space, gym.spaces.Box):
        cpu_actions = np.clip(cpu_actions, env.action_space.low, env.action_space.high)

    # Obser reward and next state
    state, rew, done, _ = env.step(cpu_actions[0])
    reward += rew
    env.render('human')

    if done:
        print ("Episode reward: ", reward)
        reward = 0.0
        state = env.reset()
        # actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
        # actor_critic.eval()

    update_current_state(state)

    # import time
    # time.sleep(0.1)
