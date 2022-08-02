import argparse
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Variable
from dataset.movi_loader import Dataset

from libs.network import KeyNet
from libs.loss import Loss
from benchmark import benchmark
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = '6pack', help = 'models from [6pack]')
parser.add_argument('--dataset', type = str, default = 'movi', help = 'dataset from [movi, ycb]')
parser.add_argument('--dataset_root', type=str, default = '/media/lang/My Passport/Dataset/MOvi', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 14,  help='category to train')
parser.add_argument('--num_pt', type=int, default = 500, help='points')
parser.add_argument('--workers', type=int, default = 30, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'ckpt/', help='save dir')
parser.add_argument('--occlude', action= 'store_true')
parser.add_argument('--ccd', action = 'store_true', help = 'Use skeleton merger to compute the CCD loss.')
parser.add_argument('--tfb', action = 'store_true', help = 'Use TF-Blender or not.')
parser.add_argument('--video_num', default = 976, type = int)
parser.add_argument('--memory_size', default=0, type = int)

opt = parser.parse_args()

model = KeyNet(opt, num_points = opt.num_pt, num_key = opt.num_kp)
model = model.float()
model.cuda()
model.eval()
criterion = Loss(opt.num_kp)
model.load_state_dict(torch.load(opt.resume))
test_dataset = Dataset(opt, mode = 'test', length = 5000, eval = True)


cls_in_5_5 = 0
cls_iou_25 = 0

cls_rot = []
cls_trans = []
cout = 0
while(test_dataset.current_video_num <= opt.video_num):
    '''
    Video
    '''
    if test_dataset.check_frame_len() == False:
        test_dataset.next_video()
    if test_dataset.index == 0:
        test_dataset.init_his()
    current_r, current_t = test_dataset.get_current_pose()
    while (True):
        try:

            img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = test_dataset.get_next(current_r, current_t)
        except:
            test_dataset.update_frame()
            continue
        break

    img, choose, cloud, anchor, scale = img.cuda(), choose.cuda(), cloud.cuda(), anchor.cuda(), scale.cuda()
    if len(test_dataset.fr_his) == 0:
        Kp_fr, att_fr = model.eval_forward(img, choose, cloud, anchor, scale, 0.0, True)
    else:
        for ind, his in enumerate(test_dataset.fr_his):
            img_feat = model.eval_forward(test_dataset.fr_his[ind], test_dataset.choose_his[ind],
                                          test_dataset.cloud_his[ind], None, None, 0.0, re_img=True, first=False)

            if ind == 0:
                feats = img_feat.unsqueeze(1)
            else:
                feats = torch.cat((feats, img_feat.unsqueeze(1)), dim=1)
        Kp_fr, att_fr = model.eval_forward(img, choose, cloud, anchor, scale, 0.0, first=False,
                                           his_feats=[feats, test_dataset.cloud_his])
        test_dataset.update_sequence(img, choose, cloud)
    min_dis = 0.0005
    while(True):
        print('Video index:', test_dataset.current_video_num, 'Frame index:', test_dataset.index)
        '''
        Per frame in the video.
        '''

        try:
            if test_dataset.update_frame():
                break
            img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = test_dataset.get_next(current_r, current_t)
            img, choose, cloud, anchor, scale = img.cuda(), choose.cuda(), cloud.cuda(), anchor.cuda(), scale.cuda()
        except:
            continue
        if len(test_dataset.fr_his) == 0:
            Kp_to, att_to = model.eval_forward(img, choose, cloud, anchor, scale, min_dis, first = False)
        else:
            for ind, his in enumerate(test_dataset.fr_his):
                img_feat = model.eval_forward(test_dataset.fr_his[ind], test_dataset.choose_his[ind], test_dataset.cloud_his[ind], None, None, min_dis, re_img= True, first=False)

                if ind == 0:
                    feats = img_feat.unsqueeze(1)
                else:
                    feats = torch.cat((feats, img_feat.unsqueeze(1)), dim = 1)

            Kp_to, att_to = model.eval_forward(img, choose, cloud, anchor, scale, min_dis, first=False, his_feats=[feats, test_dataset.cloud_his])

        test_dataset.update_sequence(img, choose, cloud)
        min_dis = 1000
        lenggth = len(Kp_to)
        for idx in range(lenggth):
            Kp_real, new_r, new_t, kp_dis, att = criterion.ev(Kp_fr[0], Kp_to[idx], att_to[idx])

            if min_dis > kp_dis:
                best_kp = Kp_to[idx]
                min_dis = kp_dis
                best_r = new_r
                best_t = new_t
                best_att = copy.deepcopy(att)
        print('Distance:', min_dis)

        current_t = current_t + np.dot(best_t, current_r.T)
        current_r = np.dot(current_r, best_r)

        c_transform = np.concatenate((current_r, current_t[:, np.newaxis]), axis = 1)
        c_transform = np.concatenate((c_transform, [[0.0, 0.0, 0.0, 1.0]]), axis = 0)
        g_transform = np.concatenate((gt_r, gt_t[:, np.newaxis]), axis = 1)
        g_transform = np.concatenate((g_transform, [[0.0, 0.0, 0.0, 1.0]]), axis = 0)
        bb3d = bb3d.transpose(1, 0)

        cm, IoU, r_er, t_er = benchmark(c_transform, g_transform, bb3d)
        cout += 1
        if r_er != None:
            cls_rot.append(r_er)
        if t_er != None:
            cls_trans.append(t_er)

        cls_in_5_5 += cm
        cls_iou_25 += IoU

        print("NEXT FRAME!!!")

score55 = cls_in_5_5 / cout
score25 = cls_iou_25 / cout
rot_error = np.mean(cls_rot)
trans_error = np.mean(cls_trans)

print('5cm 5degree:', score55 *100, 'IoU 25:', score25 * 100, 'rot error:', rot_error, 'translation error:', trans_error)



