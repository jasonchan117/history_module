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

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points, no_point_cloud):
        super(PoseNetFeat, self).__init__()
        self.no_point_cloud = no_point_cloud
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)
        if self.no_point_cloud == True:
            self.all_conv1 = torch.nn.Conv1d(256, 320, 1)
        else:
            self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        if self.no_point_cloud == True:
            emb = F.relu(self.e_conv1(emb))
            emb = F.relu(self.e_conv2(emb))
            x = F.relu(self.e_conv1(x))
            x = F.relu(self.e_conv2(x))
            x = torch.cat([emb, x], dim=1).contiguous()
            x = F.leaky_relu(self.all_conv1(x))
            x = self.all_conv2(x)
            return x
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous() #128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x

'''
Skeleton Merger Components
'''
class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy))
            ux = uy

    def forward(self, input_x):
        x = input_x
        for sublayer in self.sublayers:
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x
class Head(nn.Module):  # Decoder unit, one per line
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002)

    def forward(self, KPA, KPB):
        dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1))))
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.b_interp = 1.0 - self.f_interp
        # KPA: N x 3, KPB: N x 3
        # Interpolated: N x count x 3
        K = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + K  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb
class SkeletonMerger(nn.Module):  # Skeleton Merger structure
    def __init__(self, opt):
        super().__init__()
        self.npt = opt.num_points
        self.k = opt.num_kp
        self.DEC = nn.ModuleList()
        for i in range(self.k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(Head())
            self.DEC.append(DECN)
        self.MA = PBlock(160, 128, 64, should_perm=False)
        # self.head_lin = nn.Linear(125 * opt.num_points, 1024)
        self.MA_L = nn.Linear(64, self.k * (self.k - 1) // 2)
    def forward(self, keypoints, feat): # keypoints (bs, k, 3), feat (bs, 1, 160)

        RP = []
        L = []
        for i in range(self.k):
            for j in range(i):
                R, EM = self.DEC[i][j](keypoints[:, i, :], keypoints[:, j, :])
                RP.append(R)
                L.append(EM)
        feat = feat.view(feat.size(0), -1) # (bs, 160)
        # feat = self.head_lin(feat)
        feat = self.MA(feat)
        feat = self.MA_L(feat)
        MA = F.sigmoid(feat)
        LF = torch.cat(L, dim=1)  # P x 72 x 3
        return RP, LF, MA
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


class KeyNet(nn.Module):
    def __init__(self, num_points, num_key, num_cates, opt):
        super(KeyNet, self).__init__()
        self.record_160 = opt.record_160
        self.no_point_cloud = opt.no_point_cloud
        self.opt = opt
        self.tfb_thres = opt.tfb_thres
        self.ccd = opt.ccd
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points, self.no_point_cloud)
        self.num_cates = num_cates
        self.tfb = opt.tfb
        self.tfb_attention = opt.tfb_attention
        self.re_out = opt.relation_out
        self.sm = torch.nn.Softmax(dim=2)
        
        self.kp_1 = torch.nn.Conv1d(160, 90, 1) # input channel, output channel, kernel size
        self.kp_2 = torch.nn.Conv1d(90, 3*num_key, 1)

        self.att_1 = torch.nn.Conv1d(160, 90, 1)
        self.att_2 = torch.nn.Conv1d(90, 1, 1)

        self.sm2 = torch.nn.Softmax(dim=1)

        self.num_key = num_key

        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda().view(1, 1, 3).repeat(1, self.num_points, 1)
        if opt.ccd == True:
            self.sk = SkeletonMerger(opt)
        if opt.freeze == True:
            for p in self.parameters():
                p.requires_grad = False

        if self.tfb == True:
            self.FA = FeatureAdjustment(opt)
            self.TR = TemporalRelation(opt)
            # self.lin1 = torch.nn.Conv1d(160, 90, 1)
            # self.lin2 = torch.nn.Conv1d(90, 3 * num_key, 1)
            # self.lin1 = nn.Linear(160, 512)
            # self.lin2 = nn.Linear(512, 256)
            # self.lin3 = nn.Linear(256, 128)
            # self.lin4 = nn.Linear(128, 64)
            # self.lin5 = nn.Linear(64, 3 * num_key)
            self.lin1 = nn.Linear(160, 90)
            self.lin2 = nn.Linear(90, 64)
            self.lin3 = nn.Linear(64, 3 * num_key)


    def reanchor(self, ext_feat, min_choose):
        ext_feat = ext_feat.transpose(2, 1).contiguous()  # (1, 125, 160)
        ext_feat = ext_feat[:, min_choose, :].contiguous()  # (1, 1, 160)
        ext_feat = ext_feat.view(ext_feat.size(0), -1)  # (1, 160)
        return ext_feat
    # def forward(self, img, choose, x, anchor, scale, cate, gt_t, fra_his = None, choose_his = None, cloud_his = None, t_his = None):
    def forward(self, img, choose, x, anchor, scale, cate, gt_t, re_anchor = False, his_anchors = None):
        # his_anchors: (bs, ms, 160)

        num_anc = len(anchor[0])
        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = emb.repeat(1, 1, num_anc).contiguous()

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)
        x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
        x = (x - anchor).view(1, num_anc * self.num_points, 3).contiguous()

        x = x.transpose(2, 1).contiguous()

        if self.no_point_cloud == True:
            feat_x = self.feat(emb, emb)
        else:

            feat_x = self.feat(x, emb)


        feat_x = feat_x.transpose(2, 1).contiguous()
        # ext_feat = feat_x
        feat_x = feat_x.view(1, num_anc, self.num_points, 160).contiguous() # (1, 125, 500, 160)

        loc = x.transpose(2, 1).contiguous().view(1, num_anc, self.num_points, 3)
        weight = self.sm(-1.0 * torch.norm(loc, dim=3)).contiguous()
        weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160).contiguous()

        feat_x = torch.sum((feat_x * weight), dim=2).contiguous().view(1, num_anc, 160)
        feat_x = feat_x.transpose(2, 1).contiguous() # (1, 160, 125)
        ext_feat = feat_x

        # ext_feat = kp_feat
        scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
        output_anchor = (output_anchor * scale_anc).contiguous()
        min_choose = torch.argmin(torch.norm(output_anchor - gt_t, dim=2).view(-1))
        if re_anchor == True:
            return self.reanchor(ext_feat, min_choose)
        # Using TF-Blender
        if his_anchors != None:

            # his_anchors: (bs, ms, 160)
            # his_anchors = self.kp_3(his_anchors.transpose(2, 1).contiguous()) # (bs, 160, ms)
            # his_anchors = his_anchors.transpose(2, 1).contiguous()# (bs, ms, 160)

            # kp_feat = F.leaky_relu(self.kp_1(feat_x))# (1, 90, 125)
            # kp_feat = self.kp_2(kp_feat) # (1, 24, 125)
            # kp_feat = kp_feat.transpose(2, 1).contiguous()
            # kp_x = kp_feat.view(1, num_anc, self.num_key, 3).contiguous() # (1, 125, 8, 3)
            # kp_x = (kp_x + anchor_for_key).contiguous()
            #
            #
            #
            # min_choose = torch.argmin(torch.norm(output_anchor - gt_t, dim=2).view(-1))
            #
            # all_kp_x = kp_x.view(1, num_anc, 3*self.num_key).contiguous()
            # all_kp_x = all_kp_x[:, min_choose, :].contiguous()
            # all_kp_x = all_kp_x.view(1, self.num_key, 3).contiguous()


            current_anchor = ext_feat.transpose(2, 1).contiguous() # (1, 125, 160)
            current_anchor = current_anchor[:, min_choose, :].contiguous() # (1, 160)

            # current_anchor = current_anchor.view(current_anchor.size(0), -1)
            relations = self.TR(his_anchors, current_anchor) # (ms, bs, 160)
            if self.re_out == 'leaky' :
                relations = F.leaky_relu(relations) ## Change relu to leaky relu
            else:
                relations = F.relu(relations)
            adjustment = self.FA(his_anchors) # (ms, bs, 160)
            adjustment = F.softmax(adjustment, dim = 2)
            for index, item in enumerate(adjustment):
                if F.cosine_similarity(current_anchor, item, dim = 1) > self.tfb_thres:

                    relations[index]= 0.

            # aggregation = relations * adjustment #(ms, bs, 160)
            aggregation = adjustment * relations
            aggregation = torch.sum(aggregation, dim = 0) #(1, 160)
            # kp_feat = F.tanh(self.lin4(F.leaky_relu(self.lin3(F.leaky_relu(self.lin2(F.tanh(self.lin1(aggregation))))))))  # (1, 90)
            # kp_feat = self.lin5(kp_feat)  # (1, 24)
            # kp_feat  = self.lin2(F.tanh(self.lin1(aggregation)))
            kp_feat = self.lin3(F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(aggregation)))))
            kp_x = kp_feat.view(1, self.num_key, 3).contiguous() # (1, 8, 3)
            all_kp_x = (kp_x + anchor_for_key[:, min_choose]).contiguous()

            # feat_x = feat_x.transpose(2, 1).contiguous() #(1, 125, 160)
            # feat_x[:, min_choose] = aggregation
            # feat_x = feat_x.transpose(2, 1).contiguous() #(1, 160, 125)
        else:
            kp_feat = F.leaky_relu(self.kp_1(feat_x))# (1, 90, 125)
            kp_feat = self.kp_2(kp_feat) # (1, 24, 125)
            kp_feat = kp_feat.transpose(2, 1).contiguous()
            kp_x = kp_feat.view(1, num_anc, self.num_key, 3).contiguous() # (1, 125, 8, 3)
            kp_x = (kp_x + anchor_for_key).contiguous()



            min_choose = torch.argmin(torch.norm(output_anchor - gt_t, dim=2).view(-1))

            all_kp_x = kp_x.view(1, num_anc, 3*self.num_key).contiguous()
            all_kp_x = all_kp_x[:, min_choose, :].contiguous()
            all_kp_x = all_kp_x.view(1, self.num_key, 3).contiguous()

        scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)

        if self.record_160 == True:
            feat_160 = feat_x.transpose(2, 1).contiguous()[:, min_choose] #(1, 160)
        if his_anchors != None and self.tfb_attention == True: #Using the output from temporal relation module to boost the attention module's learning. (ms, bs, 160)
            relations = relations.transpose(1, 0).contiguous() # (bs, ms, 160)

            relations = relations.unsqueeze(1).repeat(1, 125, 1, 1).contiguous() # (bs, 125, ms, 160)

            feat_x = feat_x.transpose(2, 1).contiguous() # (1, 125, 160)
            relations = relations.sum(dim = 2) # (bs, 125, 160)
            # print(feat_x.size())
            feat_x = feat_x * relations # (bs, 125, 160)
            feat_x = feat_x.transpose(2, 1).contiguous()


        att_feat = F.leaky_relu(self.att_1(feat_x))
        att_feat = self.att_2(att_feat)
        att_feat = att_feat.transpose(2, 1).contiguous()
        att_feat = att_feat.view(1, num_anc).contiguous()
        att_x = self.sm2(att_feat).contiguous()
        # Using skeleton merger
        if self.opt.ccd == True:
            # ext_feat : (1, 125, 160)
            ext_feat = ext_feat[:, min_choose, :].contiguous() # (1, 1, 160)
            ext_feat = ext_feat.view(ext_feat.size(0), -1)
            RP, LF, MA = self.sk(all_kp_x, ext_feat)

            all_kp_x = (all_kp_x * scale_kp).contiguous()
            return all_kp_x, output_anchor, att_x, RP, LF, MA
        all_kp_x = (all_kp_x * scale_kp).contiguous()

        if self.record_160 == True:
            return all_kp_x, output_anchor, att_x, feat_160
        return all_kp_x, output_anchor, att_x

    def eval_forward(self, img, choose, ori_x, anchor, scale, space, first, re_anchor = False, his_anchors = None):

        num_anc = 125
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose)
        emb = emb.repeat(1, 1, num_anc).detach()
        #print(emb.size())

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)

        candidate_list = [-10*space, 0.0, 10*space]
        if space != 0.0:
            add_on = []
            for add_x in candidate_list:
                for add_y in candidate_list:
                    for add_z in candidate_list:
                        add_on.append([add_x, add_y, add_z])

            add_on = Variable(torch.from_numpy(np.array(add_on).astype(np.float32))).cuda().view(27, 1, 3)
        else:
            add_on = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0]).astype(np.float32))).cuda().view(1, 1, 3)

        all_kp_x = []
        all_att_choose = []
        scale_add_on = scale.view(1, 3)

        for tmp_add_on in add_on:
            tmp_add_on_scale = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_points, 1)
            tmp_add_on_key = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_key, 1)
            x = ori_x - tmp_add_on_scale

            x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
            x = (x - anchor).view(1, num_anc * self.num_points, 3)

            x = x.transpose(2, 1)
            if self.no_point_cloud == True:
                feat_x = self.feat(emb, emb)
            else:
                feat_x = self.feat(x, emb)
            feat_x = feat_x.transpose(2, 1)
            feat_x = feat_x.view(1, num_anc, self.num_points, 160).detach()

            loc = x.transpose(2, 1).view(1, num_anc, self.num_points, 3)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3))
            weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160)

            feat_x = torch.sum((feat_x * weight), dim=2).view(1, num_anc, 160)
            feat_x = feat_x.transpose(2, 1).detach()
            ext_feat = feat_x
            if re_anchor == True:
                ext_feat = ext_feat.transpose(2, 1).contiguous()  # (1, 125, 160)
                if not first:
                    att_choose = torch.argmax(att_x.view(-1))
                else:
                    att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)
                min_choose = att_choose
                ext_feat = ext_feat[:, min_choose, :].contiguous()  # (1, 1, 160)
                ext_feat = ext_feat.view(ext_feat.size(0), -1)  # (1, 160)
                return ext_feat

            if his_anchors != None and len(his_anchors[0]) > 1:
                # his_anchors: (bs, ms, 24)
                # his_anchors = self.kp_3(his_anchors.transpose(2, 1).contiguous())  # (bs, 160, ms)
                # his_anchors = his_anchors.transpose(2, 1).contiguous()  # (bs, ms, 160)

                if not first:
                    att_choose = torch.argmax(att_x.view(-1))
                else:
                    att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)

                min_choose = att_choose
                # current_anchor = self.kp_3(ext_feat).transpose(2, 1).contiguous()
                current_anchor = ext_feat.transpose(2, 1).contiguous()
                current_anchor = current_anchor[:, min_choose, :].contiguous()
                current_anchor = current_anchor.view(current_anchor.size(0), -1)
                relations = self.TR(his_anchors, current_anchor)  # (ms, bs, 160)
                if self.re_out == 'leaky':
                    relations = F.leaky_relu(relations)  ## Change relu to leaky relu
                else:
                    relations = F.relu(relations)

                adjustment = self.FA(his_anchors)  # (ms, bs, 160)
                adjustment = F.softmax(adjustment, dim = 2)
                for index, item in enumerate(adjustment):
                    if F.cosine_similarity(current_anchor, item, dim=1) > self.tfb_thres:
                        relations[index] = 0.

                aggregation = relations * adjustment  # (ms, bs, 160)
                aggregation = torch.sum(aggregation, dim=0)  # (1, 160)
                # aggregation = self.lin(aggregation)  # (1, 24)
                # kp_feat = F.leaky_relu(self.lin4(
                #     F.leaky_relu(self.lin3(F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(aggregation))))))))  # (1, 90)
                # kp_feat = self.lin5(kp_feat)  # (1, 24)
                kp_feat = self.lin3(F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(aggregation)))))
                # kp_feat = self.lin2(F.tanh(self.lin1(aggregation)))

                kp_x = kp_feat.view(1, self.num_key, 3).detach()  # (1, 8, 3)
                kp_x = (kp_x + anchor_for_key[:, min_choose]).detach()


            else:
                kp_feat = F.leaky_relu(self.kp_1(feat_x))  # (1, 90, 125)
                kp_feat = self.kp_2(kp_feat)  # (1, 24, 125)
                kp_feat = kp_feat.transpose(2, 1).contiguous()
                kp_x = kp_feat.view(1, num_anc, self.num_key, 3).contiguous()  # (1, 125, 8, 3)
                kp_x = (kp_x + anchor_for_key).detach()


            if his_anchors != None and len(his_anchors[0]) > 1 and self.tfb_attention == True:  # Using the output from temporal relation module to boost the attention module's learning. (ms, bs, 160)
                relations = relations.transpose(1, 0).contiguous()  # (bs, ms, 160)

                relations = relations.unsqueeze(1).repeat(1, 125, 1, 1).contiguous()  # (bs, 125, ms, 160)

                feat_x = feat_x.transpose(2, 1).contiguous()  # (1, 125, 160)
                relations = relations.sum(dim=2)  # (bs, 125, 160)
                # print(feat_x.size())
                feat_x = feat_x * relations  # (bs, 125, 160)
                feat_x = feat_x.transpose(2, 1).contiguous()
            att_feat = F.leaky_relu(self.att_1(feat_x))
            att_feat = self.att_2(att_feat)
            att_feat = att_feat.transpose(2, 1)
            att_feat = att_feat.view(1, num_anc)
            att_x = self.sm2(att_feat).detach()

            if not first:
                att_choose = torch.argmax(att_x.view(-1))
            else:
                att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)

            scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
            output_anchor = (output_anchor * scale_anc)

            scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)

            if his_anchors != None and len(his_anchors[0]) > 1:
                kp_x =  (kp_x.view(1, self.num_key, 3) + tmp_add_on_key).detach()
            else:

                kp_x = kp_x.view(1, num_anc, 3*self.num_key).detach()
                kp_x = (kp_x[:, att_choose, :].view(1, self.num_key, 3) + tmp_add_on_key).detach()

            kp_x = kp_x * scale_kp

            all_kp_x.append(copy.deepcopy(kp_x.detach()))
            all_att_choose.append(copy.deepcopy(att_choose.detach()))

        return all_kp_x, all_att_choose