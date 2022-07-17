# -*- coding: utf-8 -*-

import numpy as np
import torch

import utils


def calculate_acc(output, target, mask=None):
    pred = torch.max(output, dim=-1)[1]
    if mask is None:
        correct = torch.sum(torch.eq(pred, target)).item()
        return correct * 100.0 / np.prod(target.shape).item()
    else:
        correct = torch.sum(mask * torch.eq(pred, target).long()).item()
        return correct * 100.0 / torch.sum(mask).item()


def calculate_correct(output, target, mask=None):
    pred = torch.max(output, dim=-1)[1]
    if mask is None:
        correct = torch.sum(torch.eq(pred, target)).item()
    else:
        correct = torch.sum(mask * torch.eq(pred, target).long()).item()
    return correct


def JSD(p, q):
    common = utils.log(p + q)
    part1 = torch.sum(p * utils.log(2.0 * p) - p * common, dim=-1)
    part2 = torch.sum(q * utils.log(2.0 * q) - q * common, dim=-1)
    return 0.5 * part1 + 0.5 * part2


def JSD_unstable(p, q):
    m = (p + q) / 2
    return 0.5 * KLD(p, m) + 0.5 * KLD(q, m)


def KLD(p, q):
    return torch.sum(p * utils.log(p) - p * utils.log(q), dim=-1)


def L2(p, q):
    return torch.sum((p - q)**2, dim=-1)
