import argparse
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from libs.pspnet import PSPNet
import torch.distributions as tdist
import copy
from knn_cuda import KNN
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
}

class ModifiedResnet(nn.Module):

    def __init__(self):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class Pseudo3DConv(nn.Module):
    def __init__(self, opt):
        super(Pseudo3DConv, self).__init__()
        self.opt = opt
        self.knn = KNN(self.opt.topk)
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128,1)
        self.pconv1 = torch.nn.Conv1d(3, 64, 1)
        self.pconv2 = torch.nn.Conv1d(64, 128, 1)
        self.mp = nn.MaxPool1d(self.opt.topk)
        self.sm = torch.nn.Softmax(dim = 1)
        self.final_conv = torch.nn.Conv1d(256, 128, 1)
    def forward(self, img_feat, cloud):
        # img_feat:(1, 32, numpt), cloud:(1, numpt, 3)
        bs, di, _ = img_feat.size() # (1, 32, numpt)
        cloud_tar = cloud
        cloud = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud_tar = cloud_tar.transpose(2, 1).contiguous()

        _, inds = self.knn(cloud, cloud_tar) #(1, 8, numpt)
        inds = inds.transpose(2, 1).contiguous() #(1, numpt, 8)
        img_feat = img_feat.transpose(2, 1) #(1, numpt, 32)
        cloud = cloud.transpose(2, 1).contiguous() #(1, numpt, 3)
        for i, item in enumerate(inds[0]): # 1000
            selected_feat = torch.index_select(img_feat, 1, item) #(1, 8, 32)
            selected_points = torch.index_select(cloud, 1, item) #(1, 8, 3)
            current_point = cloud[0][i].view(1, 1, 3).repeat(1, self.opt.topk, 1).contiguous() #(1, 8, 3)
            weight = self.sm(-1. * torch.norm(current_point - selected_points, dim = 2)) #(1, 8)

            selected_feat = selected_feat.transpose(2, 1).contiguous()#(1, 32, 8)
            selected_feat = self.conv1(selected_feat)# (1, 64, 8)
            selected_feat = self.conv2(selected_feat)# (1, 128, 8)
            weight = weight.view(1, self.opt.topk, 1).repeat(1, 1, 128).contiguous() #(1, 8, 128)
            weight = weight.transpose(2, 1).contiguous() #(1, 128, 8)
            selected_feat = selected_feat * weight
            selected_feat = self.mp(selected_feat).view(1, 128).contiguous()
            if i == 0:
                suround_feat = selected_feat.unsqueeze(1) # (1, 1, 128)
            else:
                suround_feat = torch.cat((selected_feat, torch.index_select(img_feat, 1, item).unsqueeze(1)), dim = 1)

        # suround_feat(1, numpt, 128)

        cloud = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud = self.pconv1(cloud)
        cloud = self.pconv2(cloud) #(1, 128, numpt)
        suround_feat = suround_feat.transpose(2, 1).contiguous() #(1, 128, numpt)
        final = torch.cat((suround_feat, cloud), dim = 1) #(1, 256, numpt)
        final = self.final_conv(final) #(1, 128, numpt)
        return final

class KeypointGenerator(nn.Module):
    def __init__(self, opt):
        super(KeypointGenerator, self).__init__()
        self.opt = opt
        self.cnn = self.ModifiedResnet()
        self.fusion = Pseudo3DConv(opt)
    def forward(self, img, cloud, choose, t):

        img = self.cnn(img)
        bs, di, _, _ = img.size() # (1, 32, h, w)
        img = img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        img = torch.gather(img, 2, choose).contiguous() # (1, 32, num_pt)

        fused_feat = self.fusion(img, cloud) # (1, 128, numpt)





