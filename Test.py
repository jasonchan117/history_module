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
import matplotlib.pyplot as plt
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
parser.add_argument('--outf', type=str, default = 'diff_randomRT_0_3/', help='save dir')
parser.add_argument('--occlude', action= 'store_true')
parser.add_argument('--ccd', action = 'store_true', help = 'Use skeleton merger to compute the CCD loss.')
parser.add_argument('--tfb', action = 'store_true', help = 'Use TF-Blender or not.')
parser.add_argument('--memory_size', default=0, type = int)
parser.add_argument('--output', default = 'vis_result/diff_randomRT_0_3')
parser.add_argument('--v_id', default = 0, type = int, help = 'The video id that want to evaluate on.')
parser.add_argument('--ite', default= 1000, type = int)
parser.add_argument('--deeper', action= 'store_true', help = 'Use a deeper network.')
parser.add_argument('--d_scale', default= 10, type = float)
parser.add_argument('--mask', action = 'store_true', help = 'Using mask in the points sampled.')
parser.add_argument('--debug', action = 'store_true', help = 'help debug')
opt = parser.parse_args()
test_dataset = Dataset(opt, mode = 'test', length = 5000, eval = True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
for i, data in enumerate(testdataloader, 0):
    fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale, r_o, t_o, fr_miny, fr_maxy, fr_minx, fr_maxx, bb3d = data
    frame = np.transpose(fr_frame[0].numpy(),(1, 2, 0))
    cloud = fr_cloud[0].numpy()
    bb3d = bb3d[0].numpy()
    r = r_o[0].numpy()
    t = t_o[0].numpy()
    miny, maxy, minx, maxx = fr_miny.numpy(), fr_maxy.numpy(), fr_minx.numpy(), fr_maxx.numpy()
    test_dataset.points_vis(cloud, frame, r, t, miny, maxy, minx, maxx, bb3d)


