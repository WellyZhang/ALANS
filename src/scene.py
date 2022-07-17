# -*- coding: utf-8 -*-

from collections import OrderedDict, namedtuple
from itertools import product

import numpy as np
import torch

from utils import log, normalize

SceneProb = namedtuple("SceneProb", [
    "exist_prob", "number_prob", "type_prob", "norm_type_prob",
    "rule_type_prob", "size_prob", "norm_size_prob", "rule_size_prob",
    "color_prob", "norm_color_prob", "rule_color_prob"
])


class SceneEngine(torch.nn.Module):

    def __init__(self, number_slots):
        super(SceneEngine, self).__init__()
        self.num_slots = number_slots
        positions = list(product(range(2), repeat=self.num_slots))
        # assume nonempty
        start_index = 1
        position2number = np.sum(positions[start_index:], axis=1)
        # note the correspondence of positions: first digit from the left corresponds to part one
        positions = torch.tensor(positions[start_index:], dtype=torch.long)
        self.register_buffer("positions", positions)
        self.dim_position = self.positions.shape[0]
        self.num_pos_index_map = OrderedDict()
        for i in range(start_index, self.num_slots + 1):
            num_pos_index_map_i = torch.tensor(list(
                filter(lambda idx: position2number[idx] == i,
                       range(len(position2number)))),
                                               dtype=torch.long)
            self.num_pos_index_map[i] = num_pos_index_map_i

    def compute_scene_prob(self, exist_logprob, type_logprob, size_logprob,
                           color_logprob):
        # all in log prob
        # exist: tensor of shape (batch, 16, slot, DIM_EXIST)
        # type: tensor of shape (batch, 16, slot, DIM_TYPE)
        # size: tensor of shape (batch, 16, slot, DIM_SIZE)
        # color: tensor of shape (batch, 16, slot, DIM_COLOR)
        exist_prob = torch.exp(exist_logprob)
        position_prob, position_logprob = self.compute_position_prob(
            exist_logprob)
        number_prob = self.compute_number_prob(position_prob)
        type_prob, norm_type_prob, rule_type_prob = self.compute_type_prob(
            type_logprob, position_logprob)
        size_prob, norm_size_prob, rule_size_prob = self.compute_size_prob(
            size_logprob, position_logprob)
        color_prob, norm_color_prob, rule_color_prob = self.compute_color_prob(
            color_logprob, position_logprob)
        return SceneProb(exist_prob, number_prob, type_prob, norm_type_prob,
                         rule_type_prob, size_prob, norm_size_prob,
                         rule_size_prob, color_prob, norm_color_prob,
                         rule_color_prob)

    def compute_position_prob(self, exist_logprob):
        batch = exist_logprob.shape[0]
        exist_logprob = exist_logprob.unsqueeze(2).expand(
            -1, -1, self.dim_position, -1, -1)
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(
            batch, 16, -1, -1).unsqueeze(-1).long()
        position_logprob = torch.gather(
            exist_logprob, -1,
            index)  # (batch, 16, self.dim_position, slot, 1)
        position_logprob = torch.sum(position_logprob.squeeze(-1),
                                     dim=-1)  # (batch, 16, self.dim_position)
        position_prob = torch.exp(position_logprob)
        # assume nonempty: all-zero state is filtered out
        position_prob = normalize(position_prob)[0]
        position_logprob = log(position_prob)
        return position_prob, position_logprob

    def compute_number_prob(self, position_prob):
        all_num_prob = []
        # from 1, 2, ...
        for _, indices in self.num_pos_index_map.items():
            num_prob = torch.sum(position_prob[:, :, indices], dim=-1)
            all_num_prob.append(num_prob)
        number_prob = torch.stack(all_num_prob, dim=-1)
        return number_prob

    def compute_type_prob(self, type_logprob, position_logprob):
        batch = type_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(
            batch, 16, -1, -1).unsqueeze(-1).float()
        type_logprob = type_logprob.unsqueeze(2).expand(
            -1, -1, self.dim_position, -1, -1)
        type_logprob = index * type_logprob  # (batch, 16, self.dim_position, slot, DIM_TYPE)
        type_logprob = torch.sum(type_logprob,
                                 dim=3) + position_logprob.unsqueeze(-1)
        type_prob = torch.exp(type_logprob)
        type_prob = torch.sum(type_prob, dim=2)
        norm_type_prob = normalize(type_prob[:, :8, :])[0]
        # clamp for numerical stability
        non_inconsist_prob = torch.clamp(torch.sum(type_prob, dim=-1), max=1.0)
        rule_type_prob = torch.min(non_inconsist_prob[:, :8], dim=-1)[0]
        # rule_type_prob = torch.exp(torch.sum(log(non_inconsist_prob[:, :8]), dim=-1))
        type_prob = torch.cat(
            [type_prob, (1.0 - non_inconsist_prob).unsqueeze(-1)], dim=-1)
        return type_prob, norm_type_prob, rule_type_prob

    def compute_size_prob(self, size_logprob, position_logprob):
        batch = size_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(
            batch, 16, -1, -1).unsqueeze(-1).float()
        size_logprob = size_logprob.unsqueeze(2).expand(
            -1, -1, self.dim_position, -1, -1)
        size_logprob = index * size_logprob  # (batch, 16, self.dim_position, slot, DIM_SIZE)
        size_logprob = torch.sum(size_logprob,
                                 dim=3) + position_logprob.unsqueeze(-1)
        size_prob = torch.exp(size_logprob)
        size_prob = torch.sum(size_prob, dim=2)
        norm_size_prob = normalize(size_prob[:, :8, :])[0]
        # clamp for numerical stability
        non_inconsist_prob = torch.clamp(torch.sum(size_prob, dim=-1), max=1.0)
        rule_size_prob = torch.min(non_inconsist_prob[:, :8], dim=-1)[0]
        # rule_size_prob = torch.exp(torch.sum(log(non_inconsist_prob[:, :8]), dim=-1))
        size_prob = torch.cat(
            [size_prob, (1.0 - non_inconsist_prob).unsqueeze(-1)], dim=-1)
        return size_prob, norm_size_prob, rule_size_prob

    def compute_color_prob(self, color_logprob, position_logprob):
        batch = color_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(
            batch, 16, -1, -1).unsqueeze(-1).float()
        color_logprob = color_logprob.unsqueeze(2).expand(
            -1, -1, self.dim_position, -1, -1)
        color_logprob = index * color_logprob  # (batch, 16, self.dim_position, slot, DIM_COLOR)
        color_logprob = torch.sum(color_logprob,
                                  dim=3) + position_logprob.unsqueeze(-1)
        color_prob = torch.exp(color_logprob)
        color_prob = torch.sum(color_prob, dim=2)
        norm_color_prob = normalize(color_prob[:, :8, :])[0]
        # norm_color_prob = torch.softmax(1000 * log(color_prob[:, :8, :]), dim=-1)
        # clamp for numerical stability
        non_inconsist_prob = torch.clamp(torch.sum(color_prob, dim=-1),
                                         max=1.0)
        rule_color_prob = torch.min(non_inconsist_prob[:, :8], dim=-1)[0]
        # rule_color_prob = torch.exp(torch.sum(log(non_inconsist_prob[:, :8]), dim=-1))
        color_prob = torch.cat(
            [color_prob, (1.0 - non_inconsist_prob).unsqueeze(-1)], dim=-1)
        return color_prob, norm_color_prob, rule_color_prob
