import torch
from torch.autograd import Variable
import numpy as np

def compute_bnn_accuracy(dynamics, inputs, actions, targets, encode=False):
    acc = 0.
    for inp, act, tar in zip(inputs, actions, targets):
        # if encode:
        #     inp_var = Variable(torch.from_numpy(inp))
        #     tar_var = Variable(torch.from_numpy(tar))
        #     if args.cuda:
        #         inp_var = inp_var.cuda()
        #         tar_var = tar_var.cuda()
        #     inp_feat = actor_critic.encode(inp_var).data.cpu().numpy()
        #     # print ("Inp feat: ", inp_feat.shape)
        #     tar_feat = actor_critic.encode(tar_var).data.cpu().numpy()
        #     input_dat = np.hstack([inp_feat.reshape(inp_feat.shape[0], -1), act])
        #     # print ("inp dat:", input_dat.shape)
        #     target_dat = tar_feat.reshape(tar_feat.shape[0], -1)
        # else:
        input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
        target_dat = tar.reshape(tar.shape[0], -1)

        _out = dynamics.forward(Variable(torch.from_numpy(input_dat)).float())
        _out = _out.data.cpu().numpy()
        acc += np.mean(np.square(_out - target_dat.reshape(target_dat.shape[0], np.prod(target_dat.shape[1:])) ))
    acc /= len(inputs)
    acc /= len(inputs[0]) # per dimension squared error
    return acc

def vime_bnn_update(dynamics, inputs, actions, targets):
    pre_acc = compute_bnn_accuracy(dynamics, inputs, actions, targets)
    print ("Old BNN accuracy: ", pre_acc)
    for inp, act, tar in zip(inputs, actions, targets):
        input_dat = np.hstack([inp.reshape(inp.shape[0], -1), act])
        target_dat = tar.reshape(tar.shape[0], -1)
        dynamics.train(input_dat, target_dat)
    post_acc = compute_bnn_accuracy(dynamics, inputs, actions, targets)
    print ("New BNN accuracy: ", post_acc)
    return pre_acc, post_acc

def imle_encoding(actor_critic, inputs, actions, targets, use_cuda=False):
    inp_feats = []
    tar_feats = []
    for inp, act, tar in zip(inputs, actions, targets):
        inp_var = Variable(torch.from_numpy(inp))
        tar_var = Variable(torch.from_numpy(tar))
        if use_cuda:
            inp_var = inp_var.cuda()
            tar_var = tar_var.cuda()

        inp_feat = actor_critic.encode(inp_var).data.cpu().numpy()
        # print ("Inp feat: ", inp_feat.shape)
        tar_feat = actor_critic.encode(tar_var).data.cpu().numpy()
        inp_feats.append(inp_feat)
        tar_feats.append(tar_feat)
    return inp_feats, tar_feats

def imle_bnn_update(actor_critic, dynamics, inputs, actions, targets, use_cuda=False):
    """ Main difference is that we first compute the feature representation
    given the states and then concat the action before training """
    print ("IMLE BNN update")

    inp_feats, tar_feats = imle_encoding(actor_critic, inputs, actions, targets, use_cuda=use_cuda)
    pre_acc = compute_bnn_accuracy(dynamics, inp_feats, actions, tar_feats, encode=False)
    # pre_acc = compute_bnn_accuracy(inputs, actions, targets, encode=True)
    print ("Old BNN accuracy: ", pre_acc)
    for i in range(len(inp_feats)):
        inp_feat = inp_feats[i]
        # print ("Inp feat: ", inp_feat.shape)
        tar_feat = tar_feats[i]
        act = actions[i]
        input_dat = np.hstack([inp_feat.reshape(inp_feat.shape[0], -1), act])
        target_dat = tar_feat.reshape(tar_feat.shape[0], -1)
        dynamics.train(input_dat, target_dat, use_cuda=use_cuda)

    post_acc = compute_bnn_accuracy(dynamics, inp_feats, actions, tar_feats, encode=False)
    # post_acc = compute_bnn_accuracy(inputs, actions, targets, encode=True)
    print ("New BNN accuracy: ", post_acc)
    return pre_acc, post_acc
