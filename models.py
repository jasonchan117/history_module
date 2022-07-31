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


class Pseudo3DConv(nn.Module):
    def __init__(self, opt):
        super(Pseudo3DConv, self).__init__()
        self.opt = opt
        self.np = 8
        self.knn = KNN(self.np)
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128,1)
        self.psconv1 = torch.nn.Conv1d(128, 256, 1)
        self.psconv2 = torch.nn.Conv1d(256, 128, 1)
        self.pconv1 = torch.nn.Conv1d(3, 64, 1)
        self.pconv2 = torch.nn.Conv1d(64, 128, 1)
        self.mp = torch.nn.AvgPool1d(self.np)
        self.sm = torch.nn.Softmax(dim = 1)
        self.final_conv1 = torch.nn.Conv1d(256, 64, 1)
        self.final_conv2 = torch.nn.Conv1d(160, 64, 1)
        self.final_conv = torch.nn.Conv1d(128, 64, 1)
    def forward(self, img_feat, cloud, cloud_tar, same = False):

        # img_feat:(ms, 32, numpt), cloud:(ms, numpt, 3)
        bs, di, _ = img_feat.size() # (1, 32, numpt)

        cloud = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud_tar = cloud_tar.transpose(2, 1).contiguous()
        _, inds = self.knn(cloud_tar, cloud)  # (ms, 8, numpt)
        if same == True:
            inds_tocloud = inds
        else:
            _ , inds_tocloud = self.knn(cloud, cloud_tar)

        inds = inds.transpose(2, 1).contiguous() #(1, numpt, 8)
        inds_tocloud = inds_tocloud.transpose(2, 1).contiguous() #(1, numpt, 8)
        img_feat = img_feat.transpose(2, 1) #(1, numpt, 32)
        cloud = cloud.transpose(2, 1).contiguous() #(1, numpt, 3)
        cloud_feat = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud_feat = self.pconv1(cloud_feat)
        cloud_feat = self.pconv2(F.leaky_relu(cloud_feat)) #(1, 128, numpt)
        cloud_feat = cloud_feat.transpose(2, 1).contiguous() #(1, numpt, 128)

        selected_feat = torch.gather(img_feat.transpose(2, 1).contiguous(), 2, inds.view(1, 1, -1).repeat(1, 32, 1).contiguous())  # (1, 32, 500 * 8)

        selected_feat_point = torch.gather(cloud_feat.transpose(2, 1).contiguous(), 2, inds_tocloud.view(1, 1, -1).repeat(1, 128, 1).contiguous()) #(1, 128, 500 * 8)
        selected_points = torch.gather(cloud_tar, 2, inds.view(1, 1, -1).repeat(1, 3, 1).contiguous()) # #(1, 3, 500 * 8)
        selected_points_tocloud = torch.gather(cloud.transpose(2, 1).contiguous(), 2, inds_tocloud.view(1, 1, -1).repeat(1, 3, 1).contiguous())# #(1, 3, 500 * 8)
        current_point = cloud.view(1, 3, 1, -1).repeat(1, 1, 8, 1).contiguous() # (1, 3, 8, 500)
        current_point = current_point.transpose(3, 2).contiguous()
        current_point = current_point.view(1, 3, -1).contiguous() # (1, 3, 500 * 8)

        weight =  self.sm(-1. * torch.norm(current_point - selected_points, dim = 1).view(1, -1).contiguous()) # 1, 500 * 8

        current_point_tocloud = cloud_tar.view(1, 3, 1, -1).repeat(1, 1, 8, 1).contiguous() # (1, 3, 8, 500)
        current_point_tocloud = current_point_tocloud.transpose(3, 2).contiguous()
        current_point_tocloud = current_point_tocloud.view(1, 3, -1).contiguous() # (1, 3, 500 * 8)
        weight_tocloud = self.sm(-1. * torch.norm(current_point_tocloud - selected_points_tocloud, dim=1).view(1, -1).contiguous())

        selected_feat = self.conv1(selected_feat)  # (1, 64, 500*8)
        selected_feat = self.conv2(F.leaky_relu(selected_feat))  # (1, 128, 500*8)

        selected_feat_point = self.psconv1(selected_feat_point)
        selected_feat_point = self.psconv2(F.leaky_relu(selected_feat_point))  # (1, 128, 500*8)

        weight = weight.unsqueeze(1).repeat(1, 128, 1).contiguous()
        weight_tocloud = weight_tocloud.unsqueeze(1).repeat(1, 128, 1).contiguous()
        selected_feat = selected_feat * weight# (1, 128, 500*8)
        selected_feat_point = selected_feat_point * weight_tocloud ## (1, 128, 500*8)

        selected_feat = selected_feat.view(1, 128, 500, self.np).contiguous()
        selected_feat_point = selected_feat_point.view(1, 128, 500, self.np).contiguous()

        selected_feat = self.mp(selected_feat.view(1, 128 * 500, self.np).contiguous()).view(1, 128, 500).contiguous()
        selected_feat_point = self.mp(selected_feat_point.view(1, 128 * 500, self.np).contiguous()).view(1, 128, 500).contiguous()

        suround_feat = selected_feat
        suround_feat_p = selected_feat_point
        final1 = torch.cat((suround_feat, cloud_feat.transpose(2, 1).contiguous()), dim = 1) #(1, 256, numpt) collecting closest pixels.
        final2 = torch.cat((suround_feat_p, img_feat.transpose(2, 1).contiguous()), dim = 1) #(1, 160, numpt) collecting closest points in the point cloud
        final1 = self.final_conv1(final1) # (1, 64, numpt)
        final2 = self.final_conv2(final2) # (1, 64, numpt)
        final = torch.cat((final2, final1), dim = 1) # (1, 128, 160)
        final = self.final_conv(F.leaky_relu(final)) #(1, 64, numpt)

        return final

'''
Temporal relation module in TF-Blender
'''
class TemporalRelation(nn.Module):
    def __init__(self, opt):
        super(TemporalRelation, self).__init__()
        self.memory_size = opt.memory_size
        h_d = 128
        self.miniConv1 = torch.nn.Conv1d(4, 512, 3, padding=1)
        self.bat1 = torch.nn.BatchNorm1d(h_d)
        self.miniConv2 = torch.nn.Conv1d(512, 256, 3, padding=1)
        self.bat2 = torch.nn.BatchNorm1d(h_d)
        self.miniConv3 = torch.nn.Conv1d(256, 1, 3, padding=1)
    def g_func(self, f1, f2):
        # f1: (bs, 160)
        # Feature relation function
        a = f1 - f2
        b = f2 - f1
        return torch.cat((f1.unsqueeze(1), f2.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)), dim = 1) # (bs, 4, 160)

    def forward(self, history, current):
        # history: (bs, ms, 160) current: (bs, 160)
        history = history.transpose(1, 0).contiguous() # (ms, bs, 160)

        for i in range(len(history)):
            # print(history[i].size(), current.size())
            relation = self.g_func(history[i], current) # (bs, 4, 160)
            relation = self.miniConv1(relation) # (bs, 10, 160)
            # relation =  F.leaky_relu(relation)
            relation = self.miniConv2(relation) # (bs, 10, 160)
            # relation = F.leaky_relu(relation)
            relation = self.miniConv3(relation) # (bs, 1, 160)
            if i == 0:

                relations = relation.squeeze(1).unsqueeze(0)
            else:
                relations = torch.cat((relations, relation.squeeze(1).unsqueeze(0)), dim = 0)
        return relations # (ms, bs, 160) This is the adaptive weights between history and current frame.


'''
Feature Adjustment
'''
class FeatureAdjustment(nn.Module):
    def __init__(self, opt):
        super(FeatureAdjustment, self).__init__()
        self.memory_size = opt.memory_size
        self.W = TemporalRelation(opt)
    def forward(self, history):
        # history: (bs, ms, 160)
        history = history.transpose(1, 0).contiguous() # (ms, bs, 160)
        for i in range(len(history)):
            c = history[i] # (bs, 160)
            hi = torch.cat((history[:i], history[i + 1:]), dim=0) # (ms-1, bs, 160)

            relation = self.W(hi.transpose(1, 0).contiguous(), c) # (ms-1, bs, 160)

            for j, c_j in enumerate(relation):
                if j == 0:
                    sum_nei = (c * c_j)
                else:
                    sum_nei += (c * c_j)
            if i == 0:
                Fs = sum_nei.unsqueeze(0)
            else:
                Fs = torch.cat((Fs, sum_nei.unsqueeze(0)), dim = 0 )
        return Fs #(ms, bs, 160)


