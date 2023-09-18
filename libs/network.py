
import argparse
import os
import random
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
from models import Pseudo3DConv, PointsOp
import copy

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, opt):
        super(ModifiedResnet, self).__init__()
        if opt.deeper == True:
            self.model = psp_models['resnet34'.lower()]()
        else:
            self.model = psp_models['resnet18'.lower()]()

        # self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points, opt):
        super(PoseNetFeat, self).__init__()
        # if opt.deeper == True:
        #     mid_feat = 128
        # else:
        self.opt = opt
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        if opt.deeper == True:

            self.conv3 = torch.nn.Conv1d(128, 256, 1)
            self.conv4 = torch.nn.Conv1d(256, 320, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        if opt.deeper == True:
            self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
            self.e_conv4 = torch.nn.Conv1d(256, 320, 1)
            self.conv6 = torch.nn.Conv1d(512, 180, 1)
            self.conv7 = torch.nn.Conv1d(640, 180, 1)
            o = 1000
            s = 200
        else:
            o = 640
            s = 160
        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(o, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, s, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1) # (1, 128, 500)


        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1) # (1, 256, 500)
        if self.opt.deeper == True:
            x = F.leaky_relu(self.conv3(x))
            emb = F.leaky_relu(self.e_conv3(emb))

            pointfeat_3 = torch.cat((x, emb), dim = 1)# (1, 512, 500)
            pointfeat_3 = F.leaky_relu(self.conv6(pointfeat_3)) #(1, 180, 500)

            x = F.leaky_relu(self.conv4(x))
            emb = F.leaky_relu(self.e_conv4(emb))

            pointfeat_4 = torch.cat((x, emb), dim = 1)# (1, 640, 500)
            pointfeat_4 = F.leaky_relu(self.conv7(pointfeat_4))  # (1, 180, 500)

        x = F.relu(self.conv5(pointfeat_2))
        if self.opt.deeper == True:
            x = torch.cat([pointfeat_1, pointfeat_2, pointfeat_3, pointfeat_4, x ,], dim=1).contiguous()  # 128 + 256 + 256 + 180 + 180 = 1000

        else:
            x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous()  # 128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


class KeyNet(nn.Module):
    def __init__(self, opt, num_points, num_key):
        super(KeyNet, self).__init__()
        self.opt = opt
        self.num_points = num_points
        self.cnn = ModifiedResnet(opt)
        self.feat = PoseNetFeat(num_points, opt)
        self.feat2 = PoseNetFeat(num_points, opt)

        self.sm = torch.nn.Softmax(dim=2)

        if opt.deeper == True:
            mid_feat = 256
            self.kp_1 = torch.nn.Conv1d(200, mid_feat, 1)
            self.kp_2 = torch.nn.Conv1d(90, 3 * num_key, 1)
            self.kp_3 = torch.nn.Conv1d(mid_feat, 128, 1)
            self.kp_4 = torch.nn.Conv1d(128, 90, 1)
            self.c_f = 200
        else:
            self.kp_1 = torch.nn.Conv1d(160, 90, 1)
            self.kp_2 = torch.nn.Conv1d(90, 3 * num_key, 1)
            self.c_f = 160
        self.att_1 = torch.nn.Conv1d(self.c_f, 90, 1)
        self.att_2 = torch.nn.Conv1d(90, 1, 1)

        self.sm2 = torch.nn.Softmax(dim=1)

        self.num_key = num_key

        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda().view(1, 1, 3).repeat(
            1, self.num_points, 1)
        if self.opt.memory_size != 0 :
            self.p3d_local = Pseudo3DConv(opt)
            self.p3d_global= Pseudo3DConv(opt)

            self.diff2 = torch.nn.Conv1d( 2 * self.c_f, self.c_f, 1)

            self.diff4 = torch.nn.Conv1d(2 * self.c_f, self.c_f, 1)
    def process_his(self, his_feats, c_img, c_cloud):
        his_imgs = his_feats[0].transpose(1, 0).contiguous() # (ms, bs, 32, 500)
        his_clouds = his_feats[1]
        '''
        Global Difference
        '''
        # featc, indexc, weic = self.p3d(c_img, c_cloud, c_cloud, same=True, re_ind = True)  # (1, 64, 500)
        dens_feat_c = self.feat2(c_cloud.transpose(2, 1), c_img)
        '''
        Local Difference
        '''

        for i in range(len(his_clouds) - 1):
            # indv = len(his_clouds) - i - 1
            indv = i + 1
            if i == 0:
                dens_feat_f = self.feat2(his_clouds[indv - 1].transpose(2, 1), his_imgs[indv - 1])
            dens_feat_s = self.feat2(his_clouds[indv].transpose(2, 1), his_imgs[indv])  # (1, 160, 500)
            
            '''
            Global Difference
            '''
            dens_feat_c = self.p3d_global( c_img, c_cloud, his_imgs[i], his_clouds[i], dens_feat_c, dens_feat_f)

            dens_feat_f = self.sm2(self.p3d_local(his_imgs[indv], his_clouds[indv], his_imgs[indv - 1], his_clouds[indv - 1], dens_feat_s, dens_feat_f))

        '''
        Global Difference
        '''
        dens_feat_c = self.p3d_global(c_img, c_cloud, his_imgs[len(his_clouds)-1], his_clouds[len(his_clouds)-1],  dens_feat_c, dens_feat_f)

        '''
        Local Difference
        '''

        dens_feat_f = self.sm2(self.p3d_local(c_img, c_cloud, his_imgs[len(his_clouds)-1], his_clouds[len(his_clouds)-1], dens_feat_c, dens_feat_f))

        dens_feat = self.diff4(torch.cat((dens_feat_c, dens_feat_f), dim = 1)) # (1, 160, 500)
        return dens_feat


    def forward(self, img, choose, x, anchor = None, scale = None, gt_t = None, re_img = False, his_feats = None):
        num_anc = 125
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        if re_img == True:
            return emb
        if his_feats != None:
            c_img = emb
            c_cloud = x
            dens_feat = self.process_his(his_feats, c_img, c_cloud)
        
        emb = emb.repeat(1, 1, num_anc).contiguous()

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)
        x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
        x = (x - anchor).view(1, num_anc * self.num_points, 3).contiguous()

        x = x.transpose(2, 1).contiguous()
        feat_x = self.feat(x, emb)
        if his_feats != None:
            Fd = dens_feat.view(1, self.c_f, 1, self.num_points).repeat(1, 1, num_anc, 1).view(1, self.c_f, num_anc * self.num_points).contiguous()
            feat_x = torch.cat((feat_x, Fd), dim=1)  # (1, 320, 500 x 125)
            feat_x = self.diff2(feat_x)  # (1, 160, 500 x 125)

        feat_x = feat_x.transpose(2, 1).contiguous()
        feat_x = feat_x.view(1, num_anc, self.num_points, self.c_f).contiguous()

        loc = x.transpose(2, 1).contiguous().view(1, num_anc, self.num_points, 3)
        weight = self.sm(-1.0 * torch.norm(loc, dim=3)).contiguous()
        weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, self.c_f).contiguous()

        feat_x = torch.sum((feat_x * weight), dim=2).contiguous().view(1, num_anc, self.c_f)
        feat_x = feat_x.transpose(2, 1).contiguous()

        kp_feat = F.leaky_relu(self.kp_1(feat_x))
        if self.opt.deeper != True:
            kp_feat = self.kp_2(kp_feat)
        else:
            kp_feat = self.kp_2(F.leaky_relu(self.kp_4(F.leaky_relu(self.kp_3(kp_feat)))))
        kp_feat = kp_feat.transpose(2, 1).contiguous()
        kp_x = kp_feat.view(1, num_anc, self.num_key, 3).contiguous()
        kp_x = (kp_x + anchor_for_key).contiguous()

        att_feat = F.leaky_relu(self.att_1(feat_x))
        att_feat = self.att_2(att_feat)
        att_feat = att_feat.transpose(2, 1).contiguous()
        att_feat = att_feat.view(1, num_anc).contiguous()
        att_x = self.sm2(att_feat).contiguous()

        scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
        output_anchor = (output_anchor * scale_anc).contiguous()
        min_choose = torch.argmin(torch.norm(output_anchor - gt_t, dim=2).view(-1))

        all_kp_x = kp_x.view(1, num_anc, 3 * self.num_key).contiguous()
        all_kp_x = all_kp_x[:, min_choose, :].contiguous()
        all_kp_x = all_kp_x.view(1, self.num_key, 3).contiguous()

        scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)
        all_kp_x = (all_kp_x * scale_kp).contiguous()

        return all_kp_x, output_anchor, att_x, (out_img)

    def eval_forward(self, img, choose, ori_x, anchor = None, scale = None, space = None, first = None, re_img = False, his_feats = None):
        num_anc = 125
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)

        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose)
        if re_img == True:
            return emb
        if his_feats != None:
            c_img = emb
            c_cloud = ori_x
            dens_feat = self.process_his(his_feats, c_img, c_cloud)

        emb = emb.repeat(1, 1, num_anc).detach()


        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)

        candidate_list = [-10 * space, 0.0, 10 * space]
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
            feat_x = self.feat(x, emb)
            if his_feats != None:
                Fd = dens_feat.view(1, self.c_f, 1, self.num_points).repeat(1, 1, num_anc, 1).view(1, self.c_f,
                                                                                              num_anc * self.num_points).contiguous()
                feat_x = torch.cat((feat_x, Fd), dim=1)  # (1, 320, 500 x 125)
                feat_x = self.diff2(feat_x)  # (1, 160, 500 x 125)

            feat_x = feat_x.transpose(2, 1)
            feat_x = feat_x.view(1, num_anc, self.num_points, self.c_f).detach()

            loc = x.transpose(2, 1).view(1, num_anc, self.num_points, 3)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3))
            weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, self.c_f)

            feat_x = torch.sum((feat_x * weight), dim=2).view(1, num_anc, self.c_f)
            feat_x = feat_x.transpose(2, 1).detach()

            kp_feat = F.leaky_relu(self.kp_1(feat_x))
            if self.opt.deeper != True:
                kp_feat = self.kp_2(kp_feat)
            else:
                kp_feat = self.kp_2(F.leaky_relu(self.kp_4(F.leaky_relu(self.kp_3(kp_feat)))))
            kp_feat = kp_feat.transpose(2, 1)
            kp_x = kp_feat.view(1, num_anc, self.num_key, 3)
            kp_x = (kp_x + anchor_for_key).detach()

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
            kp_x = kp_x.view(1, num_anc, 3 * self.num_key).detach()
            kp_x = (kp_x[:, att_choose, :].view(1, self.num_key, 3) + tmp_add_on_key).detach()

            kp_x = kp_x * scale_kp

            all_kp_x.append(copy.deepcopy(kp_x.detach()))
            all_att_choose.append(copy.deepcopy(att_choose.detach()))

        return all_kp_x, all_att_choose