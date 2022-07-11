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
        self.knn = KNN(self.opt.topk)
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128,1)
        self.psconv1 = torch.nn.Conv1d(128, 256, 1)
        self.psconv2 = torch.nn.Conv1d(256, 128, 1)
        self.pconv1 = torch.nn.Conv1d(3, 64, 1)
        self.pconv2 = torch.nn.Conv1d(64, 128, 1)
        self.mp = torch.nn.MaxPool1d(self.opt.topk)
        self.sm = torch.nn.Softmax(dim = 1)
        self.final_conv1 = torch.nn.Conv1d(256, 128, 1)
        self.final_conv2 = torch.nn.Conv1d(256, 128, 1)
        self.final_conv = torch.nn.Conv1d(416, 32, 1)
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
        cloud_feat = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud_feat = self.pconv1(cloud_feat)
        cloud_feat = self.pconv2(F.leaky_relu(cloud_feat)) #(1, 128, numpt)
        cloud_feat = cloud_feat.transpose(2, 1).contiguous() #(1, numpt, 128)
        for i, item in enumerate(inds[0]): # 1000
            selected_feat = torch.index_select(img_feat, 1, item) #(1, 8, 32)
            selected_feat_point = torch.index_select(cloud_feat, 1, item) #(1, 8, 128)
            selected_feat_point = selected_feat_point.transpose(2, 1).contiguous() #(1, 128, 8)
            selected_points = torch.index_select(cloud, 1, item) #(1, 8, 3)
            current_point = cloud[0][i].view(1, 1, 3).repeat(1, self.opt.topk, 1).contiguous() #(1, 8, 3)
            weight = self.sm(-1. * torch.norm(current_point - selected_points, dim = 2)) #(1, 8)

            selected_feat = selected_feat.transpose(2, 1).contiguous()#(1, 32, 8)
            selected_feat = self.conv1(selected_feat)# (1, 64, 8)
            selected_feat = self.conv2(F.leaky_relu(selected_feat))# (1, 128, 8)

            selected_feat_point = self.psconv1(selected_feat_point)
            selected_feat_point = self.psconv2(F.leaky_relu(selected_feat_point)) #(1, 128, 8)


            weight = weight.view(1, self.opt.topk, 1).repeat(1, 1, 128).contiguous() #(1, 8, 128)
            weight = weight.transpose(2, 1).contiguous() #(1, 128, 8)
            selected_feat_point = selected_feat_point * weight
            selected_feat = selected_feat * weight
            selected_feat = self.mp(selected_feat).view(1, 128).contiguous()
            selected_feat_point = self.mp(selected_feat_point).view(1, 128).contiguous()
            if i == 0:
                suround_feat = selected_feat.unsqueeze(1) # (1, 1, 128)
                suround_feat_p = selected_feat_point.unsqueeze(1)
            else:
                suround_feat = torch.cat((selected_feat, selected_feat.unsqueeze(1)), dim = 1)
                suround_feat_p = torch.cat((selected_feat_point, selected_feat_point.unsqueeze(1)), dim = 1)
        # suround_feat(1, numpt, 128)
        # suround_feat_p (1, numpt, 128)

        suround_feat = suround_feat.transpose(2, 1).contiguous() #(1, 128, numpt)
        suround_feat_p = suround_feat_p.transpose(2, 1).contiguous()
        final1 = torch.cat((suround_feat, cloud_feat.transpose(2, 1).contiguous()), dim = 1) #(1, 256, numpt) collecting closest pixels.
        final2 = torch.cat((suround_feat_p, img_feat.transpose(2, 1).contiguous()), dim = 1) #(1, 160, numpt) collecting closest points in the point cloud
        final = torch.cat((final2, final1), dim = 1) # (1, 256, 160)
        final = self.final_conv(F.leaky_relu(final)) #(1, 32, numpt)
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
class KeypointGenerator(nn.Module):
    def __init__(self, opt):
        super(KeypointGenerator, self).__init__()
        self.opt = opt
        self.cnn = ModifiedResnet()
        self.fusion = Pseudo3DConv(opt)
        self.sm = torch.nn.Softmax(dim = 1)
        self.conv_dis1 = torch.nn.Conv1d(128, 64, 3, padding = 1)
        self.conv_dis2 = torch.nn.Conv1d(64, 32, 1)
        self.conv_dis3 = torch.nn.Conv1d(32, 1, 1)
        self.conv_seg1 = torch.nn.Conv1d(128, 64, 3, padding = 1)
        self.conv_seg2 = torch.nn.Conv1d(64, 32, 1)
        self.conv_seg3 = torch.nn.Conv1d(32, 1, 1)
        self.lin1 = nn.Linear(128, 90)
        self.lin2 = nn.Linear(90, 64)
        self.lin3 = nn.Linear(64, 3 * opt.num_kp)
    def forward(self, seg, img, cloud, choose, t):
        # t: (1, 3)
        img = self.cnn(img)
        bs, di, _, _ = img.size() # (1, 32, h, w)
        img = img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        img = torch.gather(img, 2, choose).contiguous() # (1, 32, num_pt)

        fused_feat = self.fusion(img, cloud) # (1, 128, numpt)
        seg_pre = self.conv_seg3(F.leaky_relu(self.conv_seg2(F.leaky_relu(self.conv_seg1(fused_feat))))).view(1, self.opt.num_pt).contiguous()

        dis_center = (t * -1.).view(1, 1, 3).repeat(1, self.opt.num_pt, 1).contiguous()#(1, numpt, 3)
        dis_center = self.sm(torch.norm(cloud - dis_center, 2).view(1, self.opt.num_pt).contiguous()) #(1, numpt) the farther the point from the center the higher weight
        center_weight = dis_center.unsqueeze(1).repeat(1, 128, 1).contiguous() #(1, 128, numpt)

        dis_pre = self.conv_dis3(F.leaky_relu(self.conv_dis2(F.leaky_relu(self.conv_dis1(fused_feat))))).view(1, self.opt.num_pt).contiguous() #(1, numpt)



        fused_feat *= center_weight #(1, 128, numpt)



        seg = seg.view(1, 1, self.opt.num_pt).repeat(1, 128, 1).contiguous()
        fused_feat *= seg #(1, 128, numpt)
        fused_feat = torch.sum(fused_feat, 2).view(1, 128).contiguous()

        keypoints = self.lin3(F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(fused_feat))))).view(1, self.opt.num_kp, 3) #(1, 8, 3)

        return keypoints, dis_pre, dis_center, seg_pre



