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
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = '6pack', help = 'models from [6pack]')
parser.add_argument('--dataset', type = str, default = 'movi', help = 'dataset from [movi, ycb]')
parser.add_argument('--dataset_root', type=str, default = '/media/lang/My Passport/Dataset/MOvi', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 14,  help='category to train')
parser.add_argument('--num_pt', type=int, default = 500, help='points')
parser.add_argument('--workers', type=int, default = 20, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'ckpt/', help='save dir')
parser.add_argument('--lr', default=0.0001, help='learning rate', type = float)
parser.add_argument('--occlude', action= 'store_true')
parser.add_argument('--eval_fre', default=1, type = int)
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--begin',default=0, type=int)
parser.add_argument('--ccd', action = 'store_true', help = 'Use skeleton merger to compute the CCD loss.')
parser.add_argument('--tfb', action = 'store_true', help = 'Use TF-Blender or not.')
parser.add_argument('--memory_size', default=0, type = int)
parser.add_argument('--tfb_thres', default = 0.7, type = float, help = 'The threshold to determine whether the relation to be set to 0 or not.')
parser.add_argument('--topk', default = 8 , type = int, help = 'The topk value considered in feature fusion.')
parser.add_argument('--score', default= np.Inf, type = float)
opt = parser.parse_args()
cates = ["Action Figures", "Bag", "Board Games", "Bottles and Cans and Cups", "Camera", "Car Seat", "Consumer Goods", "Hat", "Headphones", "Keyboard", "Legos", "Media Cases", "Mouse", "None", "Shoe", "Stuffed Toys", "Toys"]

models = {'6pack':KeyNet(opt, opt.num_pt, opt.num_kp)}
model = models[opt.model]
model.float()
model = model.cuda()

if opt.resume != '':

    model.load_state_dict(torch.load(opt.resume))
optimizer = optim.Adam(model.parameters(), lr = opt.lr)
criterion = Loss(opt.num_kp)
best_test = opt.score
traindataset = Dataset(opt, length=5000, mode='train')
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=opt.workers)
testdataset = Dataset(opt, length=1000, mode='test')
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=opt.workers)
for epoch in range(opt.begin, opt.epoch):

    model.train()
    train_dis_avg = 0.0
    train_count = 0

    optimizer.zero_grad()
    for i, data in enumerate(traindataloader, 0):
        print('Epoch:', epoch, 'batch:', i)
        if opt.memory_size == 0:
            fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale  = data
            fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t , to_cloud, to_choose, anchor, scale = fr_frame.cuda(), fr_r.cuda(), fr_t.cuda(), fr_cloud.cuda(), fr_choose.cuda(), to_frame.cuda(), to_r.cuda(), to_t.cuda() , to_cloud.cuda(), to_choose.cuda(), anchor.cuda(), scale.cuda()
            #print(fr_seg.shape, fr_frame.shape, fr_r.shape, fr_t.shape, fr_cloud.shape, fr_choose.shape)

            kp_fr, anc_fr, att_fr, w = model(fr_frame, fr_choose, fr_cloud, anchor, scale, fr_t)
            kp_to, anc_to, att_to, w_1 = model(to_frame, to_choose, to_cloud, anchor, scale, to_t)
        else:
            fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale, fr_his_fr, choose_his_fr, cloud_his_fr, fr_his_to, choose_his_to, cloud_his_to = data
            fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t , to_cloud, to_choose, anchor, scale = fr_frame.cuda(), fr_r.cuda(), fr_t.cuda(), fr_cloud.cuda(), fr_choose.cuda(), to_frame.cuda(), to_r.cuda(), to_t.cuda() , to_cloud.cuda(), to_choose.cuda(), anchor.cuda(), scale.cuda()

            for ind, his in enumerate(fr_his_fr):
                fr_his_fr[ind] = fr_his_fr[ind].cuda()
                choose_his_fr[ind] = choose_his_fr[ind].cuda()
                cloud_his_fr[ind] = cloud_his_fr[ind].cuda()

                fr_his_to[ind] = fr_his_to[ind].cuda()
                choose_his_to[ind] = choose_his_to[ind].cuda()
                cloud_his_to[ind] = cloud_his_to[ind].cuda()

                img_feat_fr = model(fr_his_fr[ind], choose_his_fr[ind], cloud_his_fr[ind], re_img = True)
                img_feat_to = model(fr_his_to[ind], choose_his_to[ind], cloud_his_to[ind], re_img = True)

                if ind == 0:
                    fr_feats = img_feat_fr.unsqueeze(1)
                    to_feats = img_feat_to.unsqueeze(1)
                else:
                    fr_feats = torch.cat((fr_feats, img_feat_fr.unsqueeze(1)), dim = 1)
                    to_feats = torch.cat((to_feats, img_feat_to.unsqueeze(1)), dim = 1)

            kp_fr, anc_fr, att_fr, w = model(fr_frame, fr_choose, fr_cloud, anchor, scale, fr_t, his_feats = [fr_feats, cloud_his_fr])
            kp_to, anc_to, att_to, w_1 = model(to_frame, to_choose, to_cloud, anchor, scale, to_t, his_feats = [to_feats, cloud_his_to])

        try:
            loss, _ = criterion(kp_fr, kp_to, anc_fr, anc_to, att_fr, att_to, fr_r, fr_t, to_r, to_t, scale, opt.category)
            loss.backward()
        except:
           print(w, w_1)
           sys.exit()

        train_dis_avg += loss.item()
        train_count += 1

        if train_count != 0 and train_count % 8 ==0:
            optimizer.step()
            optimizer.zero_grad()
            print(train_count, float(train_dis_avg) / 8.0)
            train_dis_avg = 0.0
        if train_count != 0 and train_count % 100 == 0:
            torch.save(model.state_dict(), '{0}/model_current_{1}.pth'.format(opt.outf, cates[opt.category]))

    if (epoch + 1) % opt.eval_fre == 0:


        optimizer.zero_grad()
        model.eval()
        score = []
        for j, data in enumerate(testdataloader, 0):
            print('index:', j)
            if opt.memory_size == 0:
                fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale = data
                fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale = fr_frame.cuda(), fr_r.cuda(), fr_t.cuda(), fr_cloud.cuda(), fr_choose.cuda(), to_frame.cuda(), to_r.cuda(), to_t.cuda(), to_cloud.cuda(), to_choose.cuda(), anchor.cuda(), scale.cuda()

                kp_fr, anc_fr, att_fr, _ = model(fr_frame, fr_choose, fr_cloud, anchor, scale, fr_t)
                kp_to, anc_to, att_to, _ = model(to_frame, to_choose, to_cloud, anchor, scale, to_t)

            else:
                fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale, fr_his_fr, choose_his_fr, cloud_his_fr, fr_his_to, choose_his_to, cloud_his_to = data
                fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose, anchor, scale = fr_frame.cuda(), fr_r.cuda(), fr_t.cuda(), fr_cloud.cuda(), fr_choose.cuda(), to_frame.cuda(), to_r.cuda(), to_t.cuda(), to_cloud.cuda(), to_choose.cuda(), anchor.cuda(), scale.cuda()

                for ind, his in enumerate(fr_his_fr):
                    fr_his_fr[ind] = fr_his_fr[ind].cuda()
                    choose_his_fr[ind] = choose_his_fr[ind].cuda()
                    cloud_his_fr[ind] = cloud_his_fr[ind].cuda()

                    fr_his_to[ind] = fr_his_to[ind].cuda()
                    choose_his_to[ind] = choose_his_to[ind].cuda()
                    cloud_his_to[ind] = cloud_his_to[ind].cuda()

                    img_feat_fr = model(fr_his_fr[ind], choose_his_fr[ind], cloud_his_fr[ind], re_img=True)
                    img_feat_to = model(fr_his_to[ind], choose_his_to[ind], cloud_his_to[ind], re_img=True)

                    if ind == 0:
                        fr_feats = img_feat_fr.unsqueeze(1)
                        to_feats = img_feat_to.unsqueeze(1)
                    else:
                        fr_feats = torch.cat((fr_feats, img_feat_fr.unsqueeze(1)), dim=1)
                        to_feats = torch.cat((to_feats, img_feat_to.unsqueeze(1)), dim=1)

                kp_fr, anc_fr, att_fr, w = model(fr_frame, fr_choose, fr_cloud, anchor, scale, fr_t,
                                                 his_feats=[fr_feats, cloud_his_fr])
                kp_to, anc_to, att_to, w_1 = model(to_frame, to_choose, to_cloud, anchor, scale, to_t,
                                                   his_feats=[to_feats, cloud_his_to])
            _, item_score = criterion(kp_fr, kp_to, anc_fr, anc_to, att_fr, att_to, fr_r, fr_t, to_r, to_t, scale,
                                      opt.category)
            print(item_score)
            score.append(item_score)
        test_dis = np.mean(np.array(score))
        print('>>>', test_dis)
        if test_dis < best_test:
            best_test = test_dis
            torch.save(model.state_dict(), '{0}/model_{1}_{2}_{3}.pth'.format(opt.outf, epoch, test_dis, cates[opt.category]))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')