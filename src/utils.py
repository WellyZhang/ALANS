# -*- coding: utf-8 -*-

import functools

import torch

from const import LOG_EPSILON


def normalize(unnorm_prob, dim=-1):
    unnorm_prob = unnorm_prob
    sum_dim = torch.sum(unnorm_prob, dim=dim, keepdim=True)
    norm_prob = unnorm_prob / sum_dim
    return norm_prob, sum_dim


def sample_action(prob, sample=True):
    if sample:
        action = torch.distributions.Categorical(prob).sample()
    else:
        action = torch.argmax(prob, dim=-1)
    logprob = torch.log(torch.gather(prob, -1,
                                     action.unsqueeze(-1))).squeeze(-1)
    return action, logprob


def log(x):
    return torch.log(x + LOG_EPSILON)


def log_softmax(x, dim=-1, mode="torch"):
    if mode == "torch":
        return torch.nn.functional.log_softmax(x, dim)
    elif mode == "utils":
        return log(torch.nn.functional.softmax(x, dim))


def batch_kronecker_product(A, B):
    # ref: https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(B.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.view(siz0 + siz1)


def squared_norm(x, dim):
    return torch.sum(x**2, dim=dim)


def rgetattr(obj, attr, *args):
    # ref: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
