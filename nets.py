# DESCRIPTION OF nets.py
#
# ネットワーク(モデル)の定義，生成を行います．

import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import metrics


class FRNet(nn.Module):
    def __init__(self, device, input_size, num_features, num_fout1, num_fout2, num_fout3):
        super(FRNet, self).__init__()
        self.device = device

        self.input_size = input_size
        self.num_fout3 = num_fout3

        # Loading to the 15th layer of trained VGG
        self.vgg16_features = nn.ModuleList(list(torchvision.models.vgg16(pretrained=True).features)[:16])
        for p in self.vgg16_features.parameters():
            p.requires_grad = False

        self.c1 = nn.Conv2d(256, num_fout1, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(num_fout1, num_fout2, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(num_fout2, num_fout3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_fout1)
        self.bn2 = nn.BatchNorm2d(num_fout2)
        # self.fc1 = nn.Linear(int(input_size/4*input_size/4*num_fout3), int(input_size/4*input_size/4*num_fout3/10) - 1)
        self.fc1 = nn.Linear(int(input_size/4*input_size/4*num_fout3), int(input_size/4*input_size/4*num_fout3/10))
        self.fc2 = nn.Linear(int(input_size/4*input_size/4*num_fout3/10), num_features)

    def forward(self, x, label=None):
        with torch.no_grad():
            self.vgg16_features.eval()
            for f in self.vgg16_features:
                x = f(x)

        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu((self.bn2(self.c2(x))))
        x = self.c3(x)
        x = x.view(-1,int(self.input_size/4*self.input_size/4*self.num_fout3))
        x = self.fc1(x)

        x = self.fc2(x)

        return x

    def concat(self, feature, label):
        # new_feature = torch.cat([feature, label], dim=1)

        x = torch.empty(feature.shape[0], feature.shape[1]+1)
        x[:, :-1] = feature
        x[:, -1] = label
        # x.to(torch.device('cuda'))

        return x.to(self.device)
