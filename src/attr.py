# -*- coding: utf-8 -*-

import math

import torch

import utils


class NumberSys(torch.nn.Module):

    def __init__(self, matrix_size, int_size, logic_size):
        super(NumberSys, self).__init__()
        self.matrix_size = matrix_size
        self.int_size = int_size
        self.logic_size = logic_size

        # int
        self.zero = torch.nn.Parameter(
            torch.Tensor(self.matrix_size, self.matrix_size))
        self.M = torch.nn.Parameter(
            torch.Tensor(self.matrix_size, self.matrix_size))

        # independent
        # self.int = torch.nn.Parameter(torch.Tensor(self.int_size, self.matrix_size, self.matrix_size))

        # logic
        # maybe require logic representation to be orthonormal (inner product of matrices is trace)
        # check vector logics: https://en.wikipedia.org/wiki/Vector_logic
        self.logic = torch.nn.Parameter(
            torch.Tensor(self.logic_size, self.matrix_size, self.matrix_size))

        self.reset_parameters()

    def reset_parameters(self):
        # init int
        torch.nn.init.normal_(self.zero)
        torch.nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))

        # independent
        # torch.nn.init.normal_(self.int)

        # init logic
        torch.nn.init.normal_(self.logic)

    def forward(self):
        int_list = [self.zero]
        for _ in range(self.int_size - 1):
            int_list.append(torch.matmul(self.M, int_list[-1]))
        # Tensor of (case, matrix_size, matrix_size)
        return torch.stack(int_list, dim=0), self.logic

        # independent
        # return self.int, self.logic


class Attribute(object):

    def __init__(self, name, space_size, mode):
        self.attr = name
        self.space_size = space_size
        self.mode = mode

    def construct(self, int_space, logic_space):
        if self.mode == "int":
            space = int_space[:self.space_size, :, :]
        elif self.mode == "logic":
            space = logic_space[:self.space_size, :, :]
        return space

    def forward(self, indices, space):
        # indices: Tensor of any shape
        # space: Tensor of shape (case, matrix_size, matrix_size)
        size = list(indices.shape)
        size.append(space.shape[-1])
        size.append(space.shape[-1])
        return torch.index_select(space, 0, indices.view(-1)).view(size)

    def decode(self, predict_matrix, space):
        # predict_matrix: Tensor of shape (batch, matrix_size, matrix_size)
        # space: Tensor of shape (case, matrix_size, matrix_size)
        predict_matrix = predict_matrix.unsqueeze(1).expand(
            -1, space.shape[0], -1, -1)
        dist = utils.squared_norm(predict_matrix - space.unsqueeze(0),
                                  dim=(2, 3))
        return torch.nn.functional.softmax(-dist, dim=-1)
