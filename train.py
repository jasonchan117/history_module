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
from libs.loss import Loss
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'movi', help = 'dataset from [movi, ycb]')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 16,  help='category to train')
parser.add_argument('--num_pt', type=int, default = 500, help='points')
parser.add_argument('--workers', type=int, default = 30, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'ckpt/', help='save dir')
parser.add_argument('--lr', default=0.0001, help='learning rate', type = float)
parser.add_argument('--occlude', action= 'store_true')
parser.add_argument('--eval_fre', default=5, type = int)
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--begin',default=0, type=int)
parser.add_argument('--ccd', action = 'store_true', help = 'Use skeleton merger to compute the CCD loss.')
parser.add_argument('--tfb', action = 'store_true', help = 'Use TF-Blender or not.')
parser.add_argument('--memory_size', default=0, type = int)
parser.add_argument('--tfb_thres', default = 0.7, type = float, help = 'The threshold to determine whether the relation to be set to 0 or not.')
parser.add_argument('--score', default= np.Inf, type = float)
opt = parser.parse_args()

model = KeypointGenerator(opt)
model.cuda()

traindataset = Dataset(opt, length = 5000, mode = 'train')
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=True, num_workers = 0)
# testdataset = Dataset(opt, length = 500, mode = 'test')
# testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True, num_workers = 0)
for i, data in enumerate(traindataloader, 0):
    fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose  = data
    print('Epoch:', i + 1)
    fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t , to_cloud, to_choose = fr_frame.cuda(), fr_r.cuda(), fr_t.cuda(), fr_cloud.cuda(), fr_choose.cuda(), to_frame.cuda(), to_r.cuda(), to_t.cuda() , to_cloud.cuda(), to_choose.cuda()
    print(fr_frame.shape, fr_r.shape, fr_t.shape, fr_cloud.shape, fr_choose.shape)