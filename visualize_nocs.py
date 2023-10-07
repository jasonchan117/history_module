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

from dataset.movi_loader import Dataset as Movi
from dataset.nocs_loader import Nocs
from libs.network import KeyNet
from libs.loss import Loss
from benchmark import benchmark
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = '6pack', help = 'models from [6pack]')
parser.add_argument('--dataset', type = str, default = 'nocs', help = 'dataset from [movi, nocs]')
parser.add_argument('--dataset_root', type=str, default = '/media/lang/My Passport/Dataset/NOCS', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 5,  help='category to train')
parser.add_argument('--num_pt', type=int, default = 500, help='points')
parser.add_argument('--workers', type=int, default = 30, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'ckpt/', help='save dir')
parser.add_argument('--occlude', action= 'store_true')
parser.add_argument('--ccd', action = 'store_true', help = 'Use skeleton merger to compute the CCD loss.')
parser.add_argument('--tfb', action = 'store_true', help = 'Use TF-Blender or not.')
parser.add_argument('--frame_num', default = 1600, type = int)
parser.add_argument('--ite', default = 0, type = int)
parser.add_argument('--deeper', action= 'store_true', help = 'Use a deeper network.')
parser.add_argument('--memory_size', default=0, type = int)
parser.add_argument('--d_scale', default= 1000, type = float)
parser.add_argument('--mask', action = 'store_true', help = 'Using mask in the points sampled.')
parser.add_argument('--debug', action = 'store_true', help = 'help debug')
parser.add_argument('--real_data', action = 'store_true', help = 'Only use real data to train.')
parser.add_argument('--output', default = 'vis_result/temp')
opt = parser.parse_args()
opt.d_scale = 10. if opt.dataset == 'movi' else 1000.
m_proj = np.array([[-280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]]) if opt.dataset == 'movi' else np.array([[591.01250, 0., 322.52500 ],[0., 590.16775, 244.11084], [0.,0.,1.]])

model = KeyNet(opt, num_points = opt.num_pt, num_key = opt.num_kp)
model = model.float()
model.cuda()
model.eval()
criterion = Loss(opt.num_kp, opt)
model.load_state_dict(torch.load(opt.resume))
test_dataset = Movi(opt, mode = 'test', length = 5000, eval = True) if opt.dataset == 'movi' else Nocs(opt, mode = 'val', length = 5000, eval = True)

while(test_dataset.check_len()):

    '''
    Video
    '''
    print('New Start')
    # if test_dataset.index == 0:
    if opt.memory_size != 0:
        test_dataset.init_his()

    choose_obj = test_dataset.real_obj_name_list[test_dataset.cate_id][test_dataset.obj_index]
    choose_frame = test_dataset.real_obj_list[test_dataset.cate_id][choose_obj][test_dataset.index]

    current_r, current_t, _ = test_dataset.get_pose(choose_frame, choose_obj)
    ################
    if opt.ite != 0:
        min_dis = 1000.
        for iterative in range(opt.ite):


            while (True):
                try:

                    img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = test_dataset.get_next(current_r, current_t, his = False)
                except:

                    test_dataset.update_frame()
                    continue
                break
            if opt.memory_size != 0:
                test_dataset.basis_rt = [current_r, current_t]
                test_dataset.basis_scale = scale.squeeze(0).numpy()
                test_dataset.re_origin( scale=scale.squeeze(0).numpy()) # Using basis rt and basis scale to process the history: back transform all the frames in history seq using curent pose and scale
                # cv2.imshow('img', np.transpose(img.numpy(), (1, 2, 0)))
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



            new_t, att, kp_dis = criterion.ev_zero(Kp_fr[0], att_fr[0])
            if min_dis > kp_dis:
                print(iterative, kp_dis)
                min_dis = kp_dis
                best_current_r = copy.deepcopy(current_r)
                best_current_t = copy.deepcopy(current_t)
                best_att = copy.deepcopy(att)

            current_t = current_t + np.dot(new_t, current_r.T)
        current_r, current_t, att = best_current_r, best_current_t, best_att

#######################################
    while (True):
        try:

            img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = test_dataset.get_next(current_r, current_t, his=False)
        except:

            test_dataset.update_frame()
            continue
        break
    if opt.memory_size != 0:
        test_dataset.basis_rt = [current_r, current_t]
        test_dataset.basis_scale = scale.squeeze(0).numpy()
        test_dataset.re_origin(scale=scale.squeeze(
            0).numpy())  # Using basis rt and basis scale to process the history: back transform all the frames in history seq using curent pose and scale
        # cv2.imshow('img', np.transpose(img.numpy(), (1, 2, 0)))
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

#################################

    if opt.memory_size != 0:
        cloud_temp = torch.from_numpy(  (test_dataset.times_scale(test_dataset.basis_scale ,( cloud.cpu().squeeze(0).numpy() * test_dataset.dis_scale)) @ test_dataset.basis_rt[0].T + test_dataset.basis_rt[1] ).astype(np.float32)).unsqueeze(0).cuda()
        test_dataset.update_sequence(img, choose, cloud_temp, scale.cpu().squeeze(0).numpy(), gt_r, gt_t)

    min_dis = 0.0005
    while(True):
        print('Object', test_dataset.real_obj_name_list[test_dataset.cate_id][test_dataset.obj_index], 'Scene:', test_dataset.scene, 'Frame: ', test_dataset.real_obj_list[test_dataset.cate_id][test_dataset.real_obj_name_list[test_dataset.cate_id][test_dataset.obj_index]][test_dataset.index])
        '''
        Per frame in the video.
        '''
        ####################
        try:
            if test_dataset.update_frame():
                break
            img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = test_dataset.get_next(current_r, current_t, his = False)
            img, choose, cloud, anchor, scale = img.cuda(), choose.cuda(), cloud.cuda(), anchor.cuda(), scale.cuda()
        except:

            continue
        if opt.memory_size != 0:
            test_dataset.basis_rt = [current_r, current_t]
            test_dataset.basis_scale = scale.cpu().squeeze(0).numpy()
            test_dataset.re_origin(scale=scale.cpu().squeeze(0).numpy())
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
        if opt.memory_size != 0:
            cloud_temp = torch.from_numpy((test_dataset.times_scale(test_dataset.basis_scale, (
                        cloud.cpu().squeeze(0).numpy() * test_dataset.dis_scale)) @ test_dataset.basis_rt[0].T +
                                           test_dataset.basis_rt[1]).astype(np.float32)).unsqueeze(0).cuda() # camera space and before scale

            test_dataset.update_sequence(img, choose, cloud_temp, scale.cpu().squeeze(0).numpy(), current_r, current_t)
        c_transform = np.concatenate((current_r, current_t[:, np.newaxis]), axis = 1)
        c_transform = np.concatenate((c_transform, [[0.0, 0.0, 0.0, 1.0]]), axis = 0)
        g_transform = np.concatenate((gt_r, gt_t[:, np.newaxis]), axis = 1)
        g_transform = np.concatenate((g_transform, [[0.0, 0.0, 0.0, 1.0]]), axis = 0)

        cam_bb3d = bb3d @ gt_r.T + gt_t
        cam_bb3d[:, 0] *=-1
        cam_bb3d[:, 1] *=-1
        cam_bb3d = (m_proj @ cam_bb3d.transpose()).transpose()
        cam_bb3d = ((1 / cam_bb3d[:, 2]) * cam_bb3d.transpose()).transpose()
        cam_bb3d = cam_bb3d.astype(np.int)
        cam_bb3d = cam_bb3d[:, 0:2]

        fig, ax = plt.subplots(1)
        ax.imshow(test_dataset.ori_img)
        x1 = [cam_bb3d[0][0], cam_bb3d[1][0], cam_bb3d[5][0], cam_bb3d[4][0]]
        y1 = [cam_bb3d[0][1], cam_bb3d[1][1], cam_bb3d[5][1], cam_bb3d[4][1]]

        x2 = [cam_bb3d[2][0], cam_bb3d[3][0], cam_bb3d[7][0], cam_bb3d[6][0]]
        y2 = [cam_bb3d[2][1], cam_bb3d[3][1], cam_bb3d[7][1], cam_bb3d[6][1]]

        x3 = [cam_bb3d[6][0], cam_bb3d[2][0], cam_bb3d[0][0], cam_bb3d[4][0], cam_bb3d[6][0]]
        y3 = [cam_bb3d[6][1], cam_bb3d[2][1], cam_bb3d[0][1], cam_bb3d[4][1], cam_bb3d[6][1]]

        x4 = [cam_bb3d[7][0], cam_bb3d[3][0], cam_bb3d[1][0], cam_bb3d[5][0], cam_bb3d[7][0]]
        y4 = [cam_bb3d[7][1], cam_bb3d[3][1], cam_bb3d[1][1], cam_bb3d[5][1], cam_bb3d[7][1]]


        
        ax.plot(x1, y1, c = 'green')
        ax.plot(x2, y2, c='green' )
        ax.plot(x3, y3, c='green')
        ax.plot(x4, y4, c='green')
        
        cam_bb3d = bb3d @ current_r.T + current_t
        cam_bb3d[:, 0] *=-1
        cam_bb3d[:, 1] *=-1
        cam_bb3d = (m_proj @ cam_bb3d.transpose()).transpose()
        cam_bb3d = ((1 / cam_bb3d[:, 2]) * cam_bb3d.transpose()).transpose()
        cam_bb3d = cam_bb3d.astype(np.int)
        cam_bb3d = cam_bb3d[:, 0:2]

        x1 = [cam_bb3d[0][0], cam_bb3d[1][0], cam_bb3d[5][0], cam_bb3d[4][0]]
        y1 = [cam_bb3d[0][1], cam_bb3d[1][1], cam_bb3d[5][1], cam_bb3d[4][1]]

        x2 = [cam_bb3d[2][0], cam_bb3d[3][0], cam_bb3d[7][0], cam_bb3d[6][0]]
        y2 = [cam_bb3d[2][1], cam_bb3d[3][1], cam_bb3d[7][1], cam_bb3d[6][1]]

        x3 = [cam_bb3d[6][0], cam_bb3d[2][0], cam_bb3d[0][0], cam_bb3d[4][0], cam_bb3d[6][0]]
        y3 = [cam_bb3d[6][1], cam_bb3d[2][1], cam_bb3d[0][1], cam_bb3d[4][1], cam_bb3d[6][1]]

        x4 = [cam_bb3d[7][0], cam_bb3d[3][0], cam_bb3d[1][0], cam_bb3d[5][0], cam_bb3d[7][0]]
        y4 = [cam_bb3d[7][1], cam_bb3d[3][1], cam_bb3d[1][1], cam_bb3d[5][1], cam_bb3d[7][1]]


        ax.plot(x1, y1, c = 'red')
        ax.plot(x2, y2, c='red' )
        ax.plot(x3, y3, c='red')
        ax.plot(x4, y4, c='red')

        plt.savefig(os.path.join(opt.output, str(choose_obj)  + '_' + str(test_dataset.index)))
        # plt.show()