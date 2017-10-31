import numpy as np
import torch
from torch.autograd import Variable

def imle_bnn_bonus(actor_critic, dynamics, obs, act, next_obs, use_cuda=False):
    """ Very similar to VIME. Look at infogain in feature space model """
    # print ("Computing VIME bonus: ", obs.shape, act.shape, next_obs.shape)
    obs = obs[np.newaxis,:]
    act = act[np.newaxis,:]
    next_obs = next_obs[np.newaxis,:]

    # unpacking var ensures gradients not passed
    obs_input = Variable(torch.from_numpy(obs)).float()
    next_obs_input = Variable(torch.from_numpy(next_obs)).float()
    if use_cuda:
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


def vime_bnn_bonus(dynamics, obs, act, next_obs):
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
