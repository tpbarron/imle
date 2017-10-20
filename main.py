import argparse
from collections import deque
from itertools import count
import csv
import copy
import os
import sys
import gym
import numpy as np
import joblib
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
torch.set_default_tensor_type('torch.FloatTensor')
from logger import Logger
# ppo imports
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from eval import run_eval_episodes
from model import *
from storage import RolloutStorage

import replay_memory
import bnn
import pybullet_envs
import gym_x

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='AcrobotContinuousVisionX-v0', help='Env to train.')
parser.add_argument('--n-iter', type=int, default=250, help='Num iters')
parser.add_argument('--max-episode-steps', type=int, default=1000, help='Max num ep steps')
parser.add_argument('--max-replay-size', type=int, default=100000, help='Max num samples to store in replay memory')
parser.add_argument('--render', action='store_true', help='Render env observations')
parser.add_argument('--vime', action='store_true', help='Do VIME update')
parser.add_argument('--imle', action='store_true', help='Do IMLE update')
parser.add_argument('--shared-actor-critic', action='store_true', help='Whether to share params between pol and val in network')

# PPO args
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
parser.add_argument('--batch-size', type=int, default=64, help='ppo batch size (default: 64)')
parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

# BNN args
parser.add_argument('--bnn-n-updates-per-step', type=int, default=500, help='num bnn updates per step')
parser.add_argument('--bnn-n-samples', type=int, default=10, help='num samples per bnn update')
parser.add_argument('--bnn-batch-size', type=int, default=32, help='batch size for bnn updates')
parser.add_argument('--bnn-lr', type=float, default=0.0001, help='lr for bnn updates')
parser.add_argument('--eta', type=float, default=0.0001, help='param balancing exploitation / explr')
parser.add_argument('--eta-decay', action='store_true', default=False, help='Whether to decay eta param')
parser.add_argument('--min-replay-size', type=int, default=500, help='Min replay size for update')
parser.add_argument('--kl-q-len', type=int, default=10, help='Past KL queue size used for normalization')

#
parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
parser.add_argument('--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')
parser.add_argument('--vis-interval', type=int, default=100, help='vis interval, one log per n updates (default: 100)')
parser.add_argument('--num-stack', type=int, default=1, help='number of frames to stack (default: 4)')
parser.add_argument('--num-frames', type=int, default=10e6, help='number of frames to train (default: 10e6)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False, help='disables visdom visualization')
parser.add_argument('--no-mean-encode', action='store_true', default=False, help='use tanh instead of mean encoding')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.vis = not args.no_vis
print (args.log_dir)
os.makedirs(args.log_dir, exist_ok=True)

assert not (args.vime and args.imle), "Cannot do both VIME and IMLE"

# some bookkeeping
log = Logger(args)
log.save_args()
log.create_csv_log()

# NOTE: in case someone is searching as I was, this wrapper will also reset the
# envs as each one finishes done
envs = SubprocVecEnv([
    make_env(args.env_name, args.seed, i, args.log_dir, args.max_episode_steps)
    for i in range(args.num_processes)
])
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# if act discrete and obssize > 1 then discrete pixels
# if act cont and obs > 1 then cont pixels
# if act

obs_shape = envs.observation_space.shape
action_shape = 1

# determine action shape
is_continuous = None
if envs.action_space.__class__.__name__ == 'Discrete':
    is_continuous = False
    action_shape = 1
else:
    is_continuous = True
    action_shape = envs.action_space.shape[0]
assert(is_continuous is not None)
# determine observation shape
if len(obs_shape) > 1: # then assume images and add frame stack
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

actor_critic = make_actor_critic(obs_shape, envs.action_space, args.shared_actor_critic, is_continuous, not args.no_mean_encode)

if args.vime:
    num_inputs = envs.observation_space.shape[0]
    num_model_inputs = num_inputs
    num_actions = envs.action_space.shape[0]
elif args.imle:
    num_inputs = envs.observation_space.shape[0]
    if len(obs_shape) > 1:
        num_model_inputs = 32 # TODO: FIXME so no automatically finds encoded shape from model
    else:
        num_model_inputs = 32
    num_actions = envs.action_space.shape[0]

if args.imle or args.vime:
    memory = replay_memory.Memory(args.max_replay_size,
                                  obs_shape, #nvs.observation_space.shape,
                                  envs.action_space.shape[0])
    # print (num_inputs, num_actions)
    dynamics = bnn.BNN(num_model_inputs+num_actions, num_model_inputs, lr=args.bnn_lr, n_samples=args.bnn_n_samples)
    kl_mean = deque(maxlen=args.kl_q_len)
    kl_std = deque(maxlen=args.kl_q_len)
    kl_previous = deque(maxlen=args.kl_q_len)

if args.cuda:
    actor_critic.cuda()
    # dynamics = dynamics.cuda()

old_model = copy.deepcopy(actor_critic)
optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)

def set_optimizer_lr(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_bnn_accuracy(inputs, actions, targets, encode=False):
    acc = 0.
    for inp, act, tar in zip(inputs, actions, targets):
        if encode:
            inp_var = Variable(torch.from_numpy(inp))
            tar_var = Variable(torch.from_numpy(tar))
            if args.cuda:
                inp_var = inp_var.cuda()
                tar_var = tar_var.cuda()
            inp_feat = actor_critic.encode(inp_var).data.cpu().numpy()
            # print ("Inp feat: ", inp_feat.shape)
            tar_feat = actor_critic.encode(tar_var).data.cpu().numpy()
            input_dat = np.hstack([inp_feat.reshape(inp_feat.shape[0], -1), act])
            # print ("inp dat:", input_dat.shape)
            target_dat = tar_feat.reshape(tar_feat.shape[0], -1)
        else:
            input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
            target_dat = tar.reshape(tar.shape[0], -1)

        _out = dynamics.forward(Variable(torch.from_numpy(input_dat)).float())
        _out = _out.data.cpu().numpy()
        acc += np.mean(np.square(_out - target_dat.reshape(target_dat.shape[0], np.prod(target_dat.shape[1:])) ))
    acc /= len(inputs)
    acc /= len(inputs[0]) # per dimension squared error
    return acc

def vime_bnn_update(inputs, actions, targets):
    pre_acc = compute_bnn_accuracy(inputs, actions, targets)
    print ("Old BNN accuracy: ", pre_acc)
    for inp, act, tar in zip(inputs, actions, targets):
        input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
        target_dat = tar.reshape(tar.shape[0], -1)
        dynamics.train(input_dat, target_dat)
    post_acc = compute_bnn_accuracy(inputs, actions, targets)
    print ("New BNN accuracy: ", post_acc)
    return pre_acc, post_acc

def vime_bnn_bonus(obs, act, next_obs):
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    obs = obs[np.newaxis,:]
    act = act[np.newaxis,:]
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    inputs = np.hstack([obs, act])
    targets = next_obs.reshape(next_obs.shape[0], -1)

    # Approximate KL by 2nd order gradient, ref VIME
    bonus = dynamics.fast_kl_div(inputs, targets)

    # Simple KL method
    # dynamics.save_old_params()
    # bonus = dynamics.kl_given_sample(inputs, targets)
    # bonus = bonus.data.cpu().numpy()
    # dynamics.reset_to_old_params()
    return bonus

def imle_bnn_update(inputs, actions, targets):
    """ Main difference is that we first compute the feature representation
    given the states and then concat the action before training """
    print ("IMLE BNN update")
    pre_acc = compute_bnn_accuracy(inputs, actions, targets, encode=True)
    print ("Old BNN accuracy: ", pre_acc)
    for inp, act, tar in zip(inputs, actions, targets):
        inp_var = Variable(torch.from_numpy(inp))
        tar_var = Variable(torch.from_numpy(tar))
        if args.cuda:
            inp_var = inp_var.cuda()
            tar_var = tar_var.cuda()

        inp_feat = actor_critic.encode(inp_var).data.cpu().numpy()
        # print ("Inp feat: ", inp_feat.shape)
        tar_feat = actor_critic.encode(tar_var).data.cpu().numpy()
        input_dat = np.hstack([inp_feat.reshape(inp_feat.shape[0], -1), act])
        # print ("inp dat:", input_dat.shape)
        target_dat = tar_feat.reshape(tar_feat.shape[0], -1)
        dynamics.train(input_dat, target_dat, use_cuda=args.cuda)
    post_acc = compute_bnn_accuracy(inputs, actions, targets, encode=True)
    print ("New BNN accuracy: ", post_acc)
    return pre_acc, post_acc

def imle_bnn_bonus(obs, act, next_obs):
    """ Very similar to VIME. Look at infogain in feature space model """
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    obs = obs[np.newaxis,:]
    act = act[np.newaxis,:]
    next_obs = next_obs[np.newaxis,:]

    # unpacking var ensures gradients not passed
    obs_input = Variable(torch.from_numpy(obs)).float()
    next_obs_input = Variable(torch.from_numpy(next_obs)).float()
    if args.cuda:
        obs_input = obs_input.cuda()
        next_obs_input = next_obs_input.cuda()

    obs_feat = actor_critic.encode(obs_input).data.cpu().numpy()
    # print ("next_obs_input: ", next_obs_input.size())
    # input("")
    next_obs_feat = actor_critic.encode(next_obs_input).data.cpu().numpy()

    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    inputs = np.hstack([obs_feat, act])
    targets = next_obs_feat.reshape(next_obs_feat.shape[0], -1)

    # Approximate KL by 2nd order gradient, ref VIME
    bonus = dynamics.fast_kl_div(inputs, targets)

    # simple KL method
    # dynamics.save_old_params()
    # bonus = dynamics.kl_given_sample(inputs, targets)
    # bonus = bonus.data.cpu().numpy()
    # dynamics.reset_to_old_params()
    return bonus

def ppo_update(num_updates, rollouts, final_rewards):
    # ppo update
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    print ("advantages: ", advantages.max(), advantages.min(), advantages.mean(), advantages.std())

    old_model.load_state_dict(actor_critic.state_dict())
    if hasattr(actor_critic, 'obs_filter'):
        old_model.obs_filter = actor_critic.obs_filter

    decayed_clip = args.clip_param * max(1.0 - float(num_updates * args.num_processes * args.num_steps) / args.num_frames, 0)

    for _ in range(args.ppo_epoch):
        sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps)), args.batch_size * args.num_processes, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)
            if args.cuda:
                indices = indices.cuda()
            states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
            actions_batch = rollouts.actions.view(-1, action_shape)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

            _, old_action_log_probs, _ = old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
            adv_targ = Variable(advantages.view(-1, 1)[indices])
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - decayed_clip, 1.0 + decayed_clip) * adv_targ
            # surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            optimizer.zero_grad()
            (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
            optimizer.step()

    rollouts.states[0].copy_(rollouts.states[-1])

    if num_updates % args.log_interval == 0:
        print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
            format(num_updates, num_updates * args.num_processes * args.num_steps,
                   final_rewards.mean(),
                   final_rewards.median(),
                   final_rewards.min(),
                   final_rewards.max(), -dist_entropy.data[0],
                   value_loss.data[0], action_loss.data[0]))
        # if final_rewards.mean() == 1.0: #Then have solved env
        #     return True, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])

    if num_updates * args.num_processes * args.num_steps > args.num_frames:
        return True, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])
    return False, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])

    # if num_updates % args.vis_interval == 0:
    #     win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)

def train():
    current_eta = args.eta
    pre_bnn_error, post_bnn_error = -1, -1
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
    current_state = torch.zeros(args.num_processes, *obs_shape)

    def update_current_state(state):
        shape_dim0 = envs.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            # print (current_state.size())
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    state = envs.reset()
    update_current_state(state)

    rollouts.states[0].copy_(current_state)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_state = current_state.cuda()
        rollouts.cuda()

    for num_update in count(1):
        step = 0
        bonuses = []

        while step < args.num_steps:
            # episode_step += 1

            # Sample actions
            # print ("data: ", rollouts.states[step].size())
            value, action = actor_critic.act(Variable(rollouts.states[step], volatile=True), encode_mean=True)
            # print ("val, act: ", value.size(), action.size())
            cpu_actions = action.data.cpu().numpy()
            if isinstance(envs.action_space, gym.spaces.Box):
                cpu_actions = np.clip(cpu_actions, envs.action_space.low, envs.action_space.high)

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            last_state = current_state.cpu().numpy()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_state.dim() == 4:
                current_state *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_state *= masks

            update_current_state(state)

            # compute bonuses
            if (args.imle or args.vime) and len(memory) >= args.min_replay_size and num_update > 0:
                bonuses = []
                for i in range(args.num_processes):
                    # TODO: if process is done, then give kl[-1]
                    bonus = 0
                    if not done[i]:
                        # compute reward bonuses
                        if args.vime:
                            bonus = vime_bnn_bonus(last_state[i], cpu_actions[i], current_state[i].cpu().numpy())
                        elif args.imle:
                            bonus = imle_bnn_bonus(last_state[i], cpu_actions[i], current_state[i].cpu().numpy())
                    else:
                        bonus = rollouts.bonuses[step-1, i][0] # previous timestep bonus, used if done
                        # print (bonus)
                    bonuses.append(bonus)
                bonuses = torch.from_numpy(np.array(bonuses)).unsqueeze(1)
            else:
                bonuses = torch.zeros(reward.size())

            if args.render:
                env.render()

            rollouts.insert(step, current_state, action.data, value.data, reward, bonuses, masks)

            if args.imle or args.vime:
                # add unnormalized bonus to memory for prioritization
                for i in range(args.num_processes):
                    memory.add_sample(current_state[i].cpu().numpy(),
                                      action[i].data.cpu().numpy(),
                                      reward[i].cpu().numpy(),
                                      1-masks[i].cpu().numpy())
                    # TODO:
                    # if 1-masks[i].numpy():
                    #     bonuses.append(bonuses[-1])
            step += 1

        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

        # if hasattr(actor_critic, 'obs_filter'):
        #     actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *obs_shape))

        raw_kls = 0.0
        scaled_kls = 0.0
        bonuses = 0.0
        if args.imle or args.vime:
            # normalize bonuses
            raw_kls = rollouts.bonuses.cpu().numpy().tolist()
            kl_previous.append(rollouts.bonuses.median())
            previous_mean_kl = np.mean(np.asarray(kl_previous))
            if previous_mean_kl > 0:
                # divide kls by previous_mean_kl
                # for i in range(len(bonuses)):
                #     bonuses[i] = bonuses[i] / previous_mean_kl
                rollouts.bonuses.div_(previous_mean_kl)
            scaled_kls = rollouts.bonuses.cpu().numpy().tolist()
            bonuses = rollouts.bonuses.mul(current_eta).cpu().numpy().tolist()
            print ("Bonuses (no eta): (mean) ", rollouts.bonuses.mean(),
                   ", (std) ", rollouts.bonuses.std(),
                   ", Bonuses (w/ eta): (mean) ", rollouts.bonuses.mul(current_eta).mean(),
                   ", (std) ", rollouts.bonuses.mul(current_eta).std())
            rollouts.apply_bonuses(current_eta)
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        do_exit, (pol_entropy, value_loss, policy_loss) = ppo_update(num_update, rollouts, final_rewards)

        # do bnn update if memory is large enough
        if (args.imle or args.vime) and memory.size >= args.min_replay_size and num_update % 1 == 0:
            print ("Updating BNN")
            obs_mean, obs_std, act_mean, act_std = memory.mean_obs_act()
            _inputss, _targetss, _actionss = [], [], []
            for _ in range(args.bnn_n_updates_per_step):
                batch = memory.sample(args.bnn_batch_size)
                obs_data = (batch['observations'] - obs_mean) / (obs_std + 1e-8)
                next_obs_data = (batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
                act_data = (batch['actions'] - act_mean) / (act_std + 1e-8)

                _inputss.append(obs_data)
                _targetss.append(next_obs_data)
                _actionss.append(act_data)

            # update bnn
            if args.vime:
                pre_bnn_error, post_bnn_error = vime_bnn_update(_inputss, _actionss, _targetss)
            elif args.imle:
                pre_bnn_error, post_bnn_error = imle_bnn_update(_inputss, _actionss, _targetss)

        try:
            replay_size = len(memory)
        except:
            replay_size = 0

        test_rewards = np.array([-1])
        if num_update % 1 == 0:
            test_rewards = run_eval_episodes(actor_critic, 1, args, obs_shape)

        log.write_row({'updates': num_update,
                    'frames': num_update * args.num_processes * args.num_steps,
                    'mean_reward': test_rewards.mean(),
                    'median_reward': np.median(test_rewards),
                    'min_reward': test_rewards.min(),
                    'max_reward': test_rewards.max(),
                    'pol_entropy': pol_entropy,
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'raw_kls': raw_kls,
                    'scaled_kls': scaled_kls,
                    'bonuses': bonuses,
                    'replay_size': replay_size,
                    'latest_pre_bnn_error': pre_bnn_error,
                    'latest_post_bnn_error': post_bnn_error})

        # if test_rewards.mean() == 1:
        #     do_exit = True

        # save model
        torch.save(actor_critic, os.path.join(args.log_dir, 'model'+str(num_update)+'.pth'))
        if (args.vime or args.imle):
            import joblib
            torch.save(dynamics, os.path.join(args.log_dir, 'bnn'+str(num_update)+'.pth'))
            joblib.dump(memory, os.path.join(args.log_dir, 'latest_memory.pkl'))

        if do_exit:
            envs.close()
            return

        frames = num_update * args.num_processes * args.num_steps
        set_optimizer_lr(args.lr * max(1.0 - frames / args.num_frames, 0))
        if args.eta_decay:
            current_eta = args.eta * max(1.0 - frames / args.num_frames, 0)


if __name__ == '__main__':
    train()
