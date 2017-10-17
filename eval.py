# Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/enjoy.py

# import argparse
import os
import gym
import gym_x
import pybullet_envs
import torch
from torch.autograd import Variable
import numpy as np

def run_eval_episodes(actor_critic, n, args, obs_shape): # env_name, num_stack, obs_shape):
    env = gym.make(args.env_name)
    # if 'Bullet' in args.env_name:
    #     env.render(mode='human')
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
        print ("Test run: ", i)
        done = False
        ep_reward = 0.
        step = 0
        state = env.reset()
        update_current_state(state)
        while not done and step < 10000: #args.max_episode_steps:
            env.render()
            value, action = actor_critic.act(Variable(current_state, volatile=True), deterministic=True)
            cpu_actions = action.data.cpu().numpy()
            if isinstance(env.action_space, gym.spaces.Box):
                cpu_actions = np.clip(cpu_actions, env.action_space.low, env.action_space.high)
            # Obs reward and next state
            state, rew, done, _ = env.step(cpu_actions[0])
            ep_reward += rew
            update_current_state(state)
            step += 1
        print ("Ep reward: ", ep_reward)
        rewards.append(ep_reward)
    env.close()
    return np.array(rewards)

if __name__ == '__main__':
    import sys
    env_name='HopperVisionBulletX-v0'
    obs_shape = (4, 32, 32)
    actor_critic = torch.load(sys.argv[1])
    actor_critic = actor_critic.cpu()
    print (actor_critic)
    actor_critic.eval()
    rewards = run_eval_episodes(actor_critic, 10, env_name, 4, obs_shape)
    print (rewards)
    # actor_critic.obs_filter.cpu()
