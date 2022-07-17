# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

import utils


class LeNet(nn.Module):

    def __init__(self, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        # x: tensor of shape (batch, 16, slot, img_size, img_size)
        # return log prob
        shape = x.shape
        x = x.view(-1, 1, 32, 32)
        out = F.softplus(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.softplus(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.softplus(self.fc1(out))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        out = utils.log_softmax(out, dim=1, mode="torch")
        return out.view(shape[0], shape[1], shape[2], -1)


class ObjectCNN(nn.Module):

    def __init__(self, exist_dim, type_dim, size_dim, color_dim):
        super(ObjectCNN, self).__init__()
        self.exist_model = LeNet(exist_dim)
        self.type_model = LeNet(type_dim)
        self.size_model = LeNet(size_dim)
        self.color_model = LeNet(color_dim)

    def forward(self, x):
        return self.exist_model(x), self.type_model(x), self.size_model(
            x), self.color_model(x)
