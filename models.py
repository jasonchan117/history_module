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

CUDA_LAUNCH_BLOCKING=1

class PointsTimes(nn.Module):
    def __init__(self, opt):
        super(PointsTimes, self).__init__()
        self.opt = opt
        self.np = 8
        self.knn = KNN(self.np)

    def forward(self, feat1, feat2, inds):
        selected_feat = torch.gather(feat2, 2,
                                     inds.view(1, 1, -1).repeat(1, 160, 1).contiguous())  # (1, 64, 500 * 8)

        current_feat = feat1.view(1, 160, 500, 1).repeat(1, 1, 1, self.np).contiguous()  # (1, 64, 500, 8)
        current_feat = current_feat.view(1, 160, -1).contiguous()
        diff = current_feat * selected_feat

        diff = diff.view(1, 160, 500, 8).contiguous()
        diff = torch.sum(diff, dim=3) / self.np  # (1 , 64, 500)

        return diff
class PointsPlus(nn.Module):
    def __init__(self, opt):
        super(PointsPlus, self).__init__()
        self.opt = opt
        self.np = 4
        self.mp = torch.nn.AvgPool1d(self.np)

    def forward(self, feat1,  feat2 , inds, dim = 64):
        
        selected_feat = torch.gather(feat2, 2, inds.view(1, 1, -1).repeat(1, dim, 1).contiguous())# (1, 64, 500 * 8)

        current_feat = feat1.view(1, dim, 500, 1).repeat(1, 1, 1, self.np).contiguous() #(1, 64, 500, 8)
        current_feat = current_feat.view(1, dim, -1).contiguous()
        diff = current_feat + selected_feat

        diff = self.mp(diff.view(1, dim* feat1.size(2), self.np).contiguous()).view(1, dim, feat1.size(2))
        # diff = torch.sum(diff, dim = 3) / self.np # (1 , 64, 500)

        return diff
class PointsDiff(nn.Module):
    def __init__(self, opt):
        super(PointsDiff, self).__init__()
        self.opt = opt
        self.np = 8
        self.knn = KNN(self.np)
    def forward(self, feat1, feat2, inds, weight):

        # inds:#(1, numpt, 8) feat: (1, 64, 500) weight: (1, 500 * 8)
        # The inds here is the closest k points from feat1 to feat2

        selected_feat = torch.gather(feat2, 2, inds.view(1, 1, -1).repeat(1, 64, 1).contiguous())# (1, 64, 500 * 8)
        current_feat = feat1.view(1, 64, 500, 1).repeat(1, 1, 1, self.np).contiguous() #(1, 64, 500, 8)
        current_feat = current_feat.view(1, 64, -1).contiguous()
        diff = current_feat - selected_feat
        weight = weight.unsqueeze(1).repeat(1, 64, 1).contiguous()
        diff*= weight
        diff = diff.view(1, 64, 500, 8).contiguous()
        diff = torch.sum(diff, dim = 3) / self.np # (1 , 64, 500)

        return diff
class PointsOp(nn.Module):
    def __init__(self, opt):
        super(PointsOp, self).__init__()
        self.opt = opt
        if opt.deeper == True:
            self.c_f = 200

        else:
            self.c_f = 160
        self.np = 8
        self.knn = KNN(self.np)
        self.pf = PointsDiff(opt)
        self.pp = PointsPlus(opt)
        self.pti = PointsTimes(opt)
        self.diff3 = torch.nn.Conv1d(self.c_f, self.c_f, 1)
        self.diff1 = torch.nn.Conv1d(64, self.c_f, 1)
    def forward(self, feat, feat1, feat2, inds, inds1, inds2, wei1, wei2, dens_feat_f, dens_feat_s):
        # dens_feat_s = feat
        pix_diff = self.pf(feat, feat1, inds1, wei1)
        pt_diff = self.pf(feat, feat2, inds1, wei1)
        diff_sum = F.sigmoid(self.diff1(self.pp(pix_diff, pt_diff, inds)))
        new_f = self.pti(dens_feat_f, diff_sum, inds2)
        # new_f = self.pti(diff_sum, dens_feat_f, inds1)
        dens_feat_f = self.diff3(self.pp(dens_feat_s, new_f, inds1, dim = self.c_f))
        return dens_feat_f
class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(160, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 1024, 1)
        self.conv3 = torch.nn.Conv1d(1024, 160, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x
class Pseudo3DConv(nn.Module):
    def __init__(self, opt):
        super(Pseudo3DConv, self).__init__()
        self.opt = opt
        self.pp = PointsPlus(opt)
        self.np = 12
        self.pnfeat = PointNetfeat()
        self.knn = KNN(self.np)
        self.knn_pp = KNN(4)
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128,1)

        self.pconv1 = torch.nn.Conv1d(3, 64, 1)
        self.pconv2 = torch.nn.Conv1d(64, 128, 1)
        self.mp = torch.nn.MaxPool1d(self.np) ################## change to Maxpool
        self.sm = torch.nn.Softmax(dim = 1)
        self.final_conv1 = torch.nn.Conv1d(256, 64, 1)
        self.final_conv2 = torch.nn.Conv1d(256, 64, 1)
        # self.final_conv = torch.nn.Conv1d(128, 64, 1)
        self.fuse_1 = torch.nn.Conv1d(128, 160, 1)
        self.fuse_2 = torch.nn.Conv1d(128, 160, 1)
    def forward(self, img, cloud, img_tar, cloud_tar, current_feat, target_feat, last = False):
        
        cloud = cloud.transpose(2, 1).contiguous()
        cloud_tar = cloud_tar.transpose(2, 1).contiguous()
        # _, inds_tar = self.knn(cloud_tar, cloud)  # (ms, 8, numpt)
        _, inds = self.knn(cloud, cloud_tar)
        # _, inds_self1 = self.knn(cloud, cloud)
        _, inds_self2 = self.knn(cloud_tar, cloud_tar)

        _, inds_pp = self.knn_pp(cloud_tar, cloud)
        cloud = cloud.transpose(2, 1).contiguous()
        cloud_tar = cloud_tar.transpose(2, 1).contiguous()

        # fuse_i_c_t, fuse_p_c_t = self.bid_diff_fuse(img, cloud, img_tar, cloud_tar, inds_self1, inds_tar) # 1, 64, 500
        fuse_i_t_c, fuse_p_t_c = self.bid_diff_fuse(img_tar, cloud_tar, img, cloud, inds_self2, inds)

        # fuse_c_t = self.fuse_1(torch.cat((fuse_p_c_t, fuse_i_c_t), dim = 1)) # 1, 160, 500
        fuse_t_c = self.fuse_2(torch.cat((fuse_p_t_c, fuse_i_t_c), dim = 1)) # 1, 160, 500
        fuse_t_c = self.pnfeat(fuse_t_c)

        # distance = torch.cosine_similarity(current_feat, target_feat) < 0.7 #1, 500
        # print(distance)
        # distance = distance.unsqueeze(1).repeat(1, 160, 1)
        
        # current_feat = current_feat * fuse_c_t * distance + current_feat * (~distance)
        target_feat = target_feat * fuse_t_c 

        

        inds_pp = inds_pp.transpose(2, 1).contiguous() 
        
        final = self.pp(current_feat, target_feat, inds_pp, dim = 160) # 1, 160, 500

        return final





        

        


    def bid_diff_fuse(self, img, cloud, img_tar, cloud_tar, inds_self, inds_tar):
        # CLOUD - CLOUD_TAR
        # img_feat:(ms, 32, numpt), cloud:(ms, numpt, 3), inds: 1, 12, 500
        # print(cloud.shape, img.shape, img_tar.shape, cloud_tar.shape, inds_self.shape, inds_tar.shape)
        bs, di, _ = img.size() # (1, 32, numpt)
        


        inds_self = inds_self.transpose(2, 1).contiguous()
        inds_tar = inds_tar.transpose(2, 1).contiguous() #(1, numpt, 8)


        ######################### Points features ###################
        
        cloud = cloud.transpose(2, 1).contiguous() #(1, 3, numpt)
        cloud_feat = self.pconv1(cloud)
        cloud_feat = self.pconv2(F.leaky_relu(cloud_feat)) #(1, 128, numpt)

        cloud_tar = cloud_tar.transpose(2, 1).contiguous()
        cloud_tar_feat = self.pconv1(cloud_tar)
        cloud_tar_feat = self.pconv2(F.leaky_relu(cloud_tar_feat))#(1, 128, numpt)

        ######################### Selected image features ###################

        selected_img_tar_feat = torch.gather(img_tar, 2, inds_tar.view(1, 1, -1).repeat(1, 32, 1).contiguous()) # 1, 32, numpt* 8

        ######################## Selected points features ###################

        selected_points_tar_feat = torch.gather(cloud_tar_feat, 2, inds_tar.view(1, 1, -1).repeat(1, 128, 1).contiguous()) # 1, 128, numpt* 8

        ######################## Weight computing ##########################
        current_point = cloud.view(1, 3, 500, 1).repeat(1, 1,  1, self.np).contiguous().view(1, 3, -1).contiguous() # (1, 3, 500* 8)

        selected_points_tar = torch.gather(cloud_tar, 2, inds_tar.view(1, 1, -1).repeat(1, 3, 1).contiguous()) # #(1, 3, 500 * 8)
        
        selected_points_self = torch.gather(cloud, 2, inds_self.view(1, 1, -1).repeat(1, 3, 1).contiguous())# #(1, 3, 500 * 8)

        weight_c_t = self.sm(-1. * torch.norm(current_point - selected_points_tar, dim = 1)).view(1, cloud.size(2) , self.np).transpose(2, 1).contiguous() # 1, 8, num_pt

        ###################### Weighted summation ############################



        selected_img_tar_feat = self.conv1(selected_img_tar_feat)
        selected_img_tar_feat = self.conv2(F.leaky_relu(selected_img_tar_feat)) # 1, 128 500* 8



        weight_c_t = weight_c_t.transpose(2, 1).contiguous()
        weight_c_t = weight_c_t.view(1, -1).contiguous()
        weight_c_t = weight_c_t.unsqueeze(1).repeat(1, 128, 1).contiguous() # 1, 128, numpt* 8

        selected_points_tar_feat = selected_points_tar_feat * weight_c_t
        selected_img_tar_feat = selected_img_tar_feat * weight_c_t


        selected_points_tar_feat = self.mp(selected_points_tar_feat.view(1, 128 * 500, self.np).contiguous()).view(1, 128, 500).contiguous()
        selected_img_tar_feat = self.mp(selected_img_tar_feat.view(1, 128 * 500, self.np).contiguous()).view(1, 128, 500).contiguous()



        img_feat = self.conv1(img)
        img_feat = self.conv2(F.leaky_relu(img_feat)) # 1, 128, 500
        

        ########################### feature difference ####################



        img_diff = img_feat - selected_img_tar_feat # 1, 128, 500
        cloud_diff = cloud_feat - selected_points_tar_feat
 

        ########################## Bidirectional difference fusion #############

        selected_points_self_diff = torch.gather(cloud_diff, 2, inds_self.view(1, 1, -1).repeat(1, 128, 1).contiguous())# (1, 128, 500 * 8)
        selected_img_self_diff = torch.gather(img_diff, 2, inds_self.view(1, 1, -1).repeat(1, 128, 1).contiguous())
        

        weight_c_c = self.sm(-1. * torch.norm(current_point - selected_points_self, dim = 1)).view(1, cloud.size(2) , self.np).transpose(2, 1).contiguous() # 1, 8, 500
        weight_c_c = weight_c_c.transpose(2, 1).contiguous()
        weight_c_c = weight_c_c.view(1, -1).contiguous()
        weight_c_c = weight_c_c.unsqueeze(1).repeat(1, 128, 1) # 1, 128, 500 * 8

        selected_img_self_diff = selected_img_self_diff * weight_c_c
        selected_points_self_diff = selected_points_self_diff * weight_c_c


        selected_points_self_diff = selected_points_self_diff.view(1, 128* cloud.size(2),self.np).contiguous()
        selected_img_self_diff = selected_img_self_diff.view(1, 128* cloud.size(2), self.np).contiguous()# 1, 128* 500, 8

        selected_points_self_diff = self.mp(selected_points_self_diff).view(1, 128, cloud.size(2)).contiguous()
        selected_img_self_diff = self.mp(selected_img_self_diff).view(1, 128, cloud.size(2)).contiguous() # 1, 128, 500


        fuse_i = torch.cat((img_diff, selected_points_self_diff), dim = 1).contiguous() # 1, 256, 500

        fuse_p = torch.cat((cloud_diff, selected_img_self_diff), dim = 1).contiguous()
 
        fuse_i = self.final_conv1(fuse_i)
        fuse_p = self.final_conv2(fuse_p)

        return fuse_i, fuse_p

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


