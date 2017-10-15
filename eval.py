# Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/enjoy.py

# import argparse
import os
import gym
import gym_x
import torch
from torch.autograd import Variable
import numpy as np
# import model
#
# parser = argparse.ArgumentParser(description='RL')
# parser.add_argument('--seed', type=int, default=1,
#                     help='random seed (default: 1)')
# parser.add_argument('--num-stack', type=int, default=4,
#                     help='number of frames to stack (default: 4)')
# parser.add_argument('--log-interval', type=int, default=10,
#                     help='log interval, one log per n updates (default: 10)')
# parser.add_argument('--env-name', default='PongNoFrameskip-v4',
#                     help='environment to train on (default: PongNoFrameskip-v4)')
# parser.add_argument('--load-path', default='./trained_models/',
#                     help='directory to save agent logs (default: ./trained_models/)')
# parser.add_argument('--log-dir', default='/tmp/gym/',
#                     help='directory to save agent logs (default: /tmp/gym)')
# args = parser.parse_args()
#
# try:
#     os.makedirs(args.log_dir)
# except OSError:
#     pass
#
# env = gym.make(args.env_name) #make_env(args.env_name, args.seed, 0, args.log_dir)()
# env.seed(args.seed)
#
# obs_shape = env.observation_space.shape
# obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
# current_state = torch.zeros(1, *obs_shape)
#
# actor_critic = torch.load(args.load_path)
# print (actor_critic)
# actor_critic.eval()
# actor_critic = actor_critic.cpu()
# actor_critic.obs_filter.cpu()
# def update_current_state(state):
#     shape_dim0 = env.observation_space.shape[0]
#     state = torch.from_numpy(state).float()
#     if args.num_stack > 1:
#         current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
#     current_state[:, -shape_dim0:] = state
#
# env.render('human')
# state = env.reset()
# update_current_state(state)
#
# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "torso"):
#             torsoId = i
#
# while True:
#     value, action = actor_critic.act(Variable(current_state, volatile=True)) #, deterministic=True)
#     cpu_actions = action.data.cpu().numpy()
#
#     # Obser reward and next state
#     state, _, done, _ = env.step(cpu_actions[0])
#
#     if args.env_name.find('Bullet') > -1:
#         if torsoId > -1:
#             distance = 5
#             yaw = 0
#             humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
#             p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
#
#     env.render('human')
#
#     if done:
#         state = env.reset()
#         # actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
#         # actor_critic.eval()
#
#     update_current_state(state)


def run_eval_episodes(actor_critic, n, args, obs_shape):
    env = gym.make(args.env_name)
    current_state = torch.zeros(1, *obs_shape)

    def update_current_state(state):
        shape_dim0 = env.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    if args.cuda:
        current_state = current_state.cuda()

    rewards = []
    for i in range(n):
        done = False
        ep_reward = 0.
        step = 0
        state = env.reset()
        update_current_state(state)
        while not done and step < args.max_episode_steps:
            env.render()
            value, action = actor_critic.act(Variable(current_state, volatile=True), deterministic=False)
            cpu_actions = action.data.cpu().numpy()
            if isinstance(env.action_space, gym.spaces.Box):
                cpu_actions = np.clip(cpu_actions, env.action_space.low, env.action_space.high)
            # Obs reward and next state
            state, rew, done, _ = env.step(cpu_actions[0])
            ep_reward += rew
            update_current_state(state)
            step += 1
        rewards.append(ep_reward)
    env.close()
    return np.array(rewards)
