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
from models import KeypointGenerator
from libs.network import KeyNet
from libs.loss import Loss
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
parser.add_argument('--video_num', default = 976)
parser.add_argument('--memory_size', default=0, type = int)

opt = parser.parse_args()

model = KeyNet(num_points = opt.num_pt, num_key = opt.num_kp)
model = model.float()
model.cuda()
model.eval()
criterion = Loss(opt.num_kp)
model.load_state_dict(torch.load(opt.resume))
test_dataset = Dataset(opt, mode = 'test', length = 5000, eval = True)

while(test_dataset.current_video_num <= opt.video_num):
    '''
    Video
    '''

    current_r, current_t = test_dataset.get_current_pose()
    img, choose, cloud, anchor, scale = test_dataset.get_next(current_r, current_t)
    img, choose, cloud, anchor, scale = img.cuda(), choose.cuda(), cloud.cuda(), anchor.cuda(), scale.cuda()
    Kp_fr, att_fr = model.eval_forward(img, choose, cloud, anchor, scale, 0.0, True)
    min_dis = 0.0005
    while(True):
        print('Video index:', test_dataset.current_video_num, 'Frame index:', test_dataset.index)
        '''
        Per frame in the video.
        '''
        try:
            if test_dataset.update_frame():
                break

            img, choose, cloud, anchor, scale = test_dataset.get_next(current_r, current_t)
            img, choose, cloud, anchor, scale = img.cuda(), choose.cuda(), cloud.cuda(), anchor.cuda(), scale.cuda()
        except:
            continue
        Kp_to, att_to = model.eval_forward(img, choose, cloud, anchor, scale, min_dis, False)

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
        print(min_dis)

        current_t = current_t + np.dot(best_t, current_r.T)
        current_r = np.dot(current_r, best_r)





        print("NEXT FRAME!!!")




