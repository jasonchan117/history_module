import copy

import seaborn as sns
import matplotlib
import matplotlib.colors
import numpy as np
import mediapy as media
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import PIL
from libs.transformations import euler_matrix
import torchvision.transforms as transforms
import io
import sys
import torch.utils.data as data
from PIL import Image
import os
import os.path
import cv2 as cv
import random
import torch
import matplotlib.patches
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML as html_print
import matplotlib.pyplot as plt
import math
import quaternion
c = ["Action Figures", "Bag", "Board Games", "Bottles and Cans and Cups", "Camera", "Car Seat", "Consumer Goods", "Hat", "Headphones", "Keyboard", "Legos", "Media Cases", "Mouse", "None", "Shoe", "Stuffed Toys", "Toys"]



class Dataset(data.Dataset):
    def __init__(self, opt, mode = 'train', length = 5000, eval = False):
        self.ms = opt.memory_size
        self.padding = 1
        self.eval = eval
        self.mask = opt.mask
        if eval == True:
            self.e_scale = 1.
        else:
            self.e_scale = 1.
        self.num_pt = opt.num_pt
        self.opt = opt
        self.debug = opt.debug
        self.root = opt.dataset_root
        self.mode = mode
        self.video_num = len(os.listdir(os.path.join(opt.dataset_root, mode, c[opt.category])))
        self.length = length
        self.dis_scale = opt.d_scale
        self.cate = opt.category
        self.intrinsics = np.array([[280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        self.border_list = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280]
        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.eval == True:
            # History sequence
            self.fr_his = []
            self.choose_his = []
            self.cloud_his = []

            self.current_video_num = 0
            self.index = 0
            self.video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(self.current_video_num))
            category = np.load(os.path.join(self.video_path, 'category.npy'))
            self.obj_index = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]

            # visible_sequence = np.load(os.path.join(video_path, 'bbox_frames_n.npy'), allow_pickle=True)[self.obj_index]
            # current_frame_index = visible_sequence[self.seq_index]

    def check_frame_len(self):
        visible_sequence = np.load(os.path.join(self.video_path, 'bbox_frames_n.npy'), allow_pickle=True)[
            self.obj_index]

        if len(visible_sequence) <= self.opt.memory_size :
            return False
        return True
    def next_video(self, video_num = None):
        self.fr_his = []
        self.choose_his = []
        self.cloud_his = []

        self.index = 0
        if video_num == None:
            self.current_video_num += 1
        else:
            self.current_video_num = video_num
        self.video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category],
                                       str(self.current_video_num))
        category = np.load(os.path.join(self.video_path, 'category.npy'))
        self.obj_index = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]
    def re_origin(self, scale = [1., 1.,1], only_scale = False):
        if only_scale == True:
            for i in range(len(self.cloud_his)):
                self.cloud_his[i] = torch.from_numpy(self.divide_scale(scale , self.cloud_his[i].cpu().squeeze(0).numpy()).astype(np.float32)).unsqueeze(0).cuda()
        else:
            # print(len(self.cloud_his))
            for i in range(len(self.cloud_his)):
                self.cloud_his[i] = torch.from_numpy( self.divide_scale(scale ,((self.cloud_his[i].cpu().squeeze(0).numpy() - self.basis_rt[1]) @ self.basis_rt[0]) / self.dis_scale).astype(np.float32)   ).unsqueeze(0).cuda()
    def back_cam(self):
        for i in range(self.opt.memory_size):
            self.cloud_his[i] =  torch.from_numpy(  (self.times_scale(self.basis_scale ,( self.cloud_his[i].cpu().squeeze(0).numpy() * self.dis_scale)) @ self.basis_rt[0].T + self.basis_rt[1] ).astype(np.float32)).unsqueeze(0).cuda()
    def init_his(self):
        self.fr_his = []
        self.choose_his = []
        self.cloud_his = []
        for i in range(self.opt.memory_size):
            while(True):
                r, t = self.get_current_pose()
                try:
                    img, choose, cloud, anchor, scale, gt_r, gt_t, bb3d = self.get_next(r, t, his = True)
                    cloud = cloud @ r.T + t # cam space
                except:
                    self.update_frame()
                    continue
                self.fr_his.append(img.cuda())
                self.choose_his.append(choose.cuda())
                self.cloud_his.append(cloud)
                self.update_frame()

                break

        # self.re_origin()



    def update_sequence(self, frame, choose, cloud, scale, r, t):
        self.back_cam() # his back to cam space and scales back according to basis and dis_scale back
        # self.basis_rt = [r, t] # update basis rt

        self.fr_his.append(frame)
        self.choose_his.append(choose)
        self.cloud_his.append(cloud)
        if len(self.fr_his) > self.opt.memory_size:

            self.fr_his.pop(0)
            self.choose_his.pop(0)
            self.cloud_his.pop(0)
        # self.re_origin() # Re origin the history according to the last frame in history
        # self.basis_scale = scale
    def update_frame(self):
        self.index += 1
        visible_sequence = np.load(os.path.join(self.video_path, 'bbox_frames_n.npy'), allow_pickle=True)[self.obj_index]

        if self.index >= len(visible_sequence): # Next video
            self.fr_his = []
            self.choose_his = []
            self.cloud_his = []


            self.index = 0
            self.current_video_num += 1
            self.video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(self.current_video_num))
            category = np.load(os.path.join(self.video_path, 'category.npy'))
            self.obj_index = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]
            return True
        return False
    def get_current_pose(self):
        visible_sequence = np.load(os.path.join(self.video_path, 'bbox_frames_n.npy'), allow_pickle=True)[self.obj_index]
        # pcam = (bb3d-c_t) @ c_r
        # pobj = (bb3d-t) @ r
        # print('----')
        # print(pcam, pobj @ r.T @ c_r + c_r.T @ (t - c_t))
        r, t = self.get_pose(self.obj_index, visible_sequence[self.index], self.video_path)
        c_r = np.load(os.path.join(self.video_path, 'cam_r_' + str(visible_sequence[self.index]) + '.npy'))
        c_t = np.load(os.path.join(self.video_path, 'cam_t_' + str(visible_sequence[self.index]) + '.npy'))
        return (r.T @ c_r).T, c_r.T @ (t - c_t) # No need to do the scaling.
    def get_next(self, r, t, full_img = False, his = False):
        visible_sequence = np.load(os.path.join(self.video_path, 'bbox_frames_n.npy'), allow_pickle=True)[self.obj_index]
        # Bb3d is in object space
        bb3d, img, depth, miny, maxy, minx, maxx = self.get_frame(visible_sequence[self.index], self.video_path, self.obj_index, eval = True, current_r=r, current_t=t, full_img = full_img)
        limit = self.search_fit(bb3d)
        bb3d = bb3d / self.dis_scale
        anchor_box, scale = self.get_anchor_box(bb3d)

        w_r, w_t = self.get_pose(self.obj_index, visible_sequence[self.index], self.video_path)
        c_r = np.load(os.path.join(self.video_path, 'cam_r_' + str(visible_sequence[self.index]) + '.npy'))
        c_t = np.load(os.path.join(self.video_path, 'cam_t_' + str(visible_sequence[self.index]) + '.npy'))

        gt_r, gt_t = (w_r.T @ c_r).T, c_r.T @ (w_t - c_t) # Ground truth rotation and translation from object space to camera space

        bb3d = np.load(os.path.join(self.video_path, 'bboxes_3d.npy'))[self.obj_index][visible_sequence[self.index]]

        bb3d = (bb3d - w_t) @ w_r # Object space

        # Object space
        cloud, choose = self.get_cloud(depth, miny, maxy, minx, maxx , self.video_path, visible_sequence[self.index], limit, eval = True, current_r=r, current_t=t, cate_in = self.obj_index )
        if his != True:
            cloud = cloud / self.dis_scale
            cloud, _ = self.change_to_scale(scale, cloud_fr = cloud, eval = True)

        if full_img == True:
            return self.norm(torch.from_numpy(img[0].astype(np.float32))).unsqueeze(0), \
                   torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
                   torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
                   torch.from_numpy(anchor_box.astype(np.float32)).unsqueeze(0), \
                   torch.from_numpy(scale.astype(np.float32)).unsqueeze(0), gt_r, gt_t, bb3d, img[1]
        return self.norm(torch.from_numpy(img.astype(np.float32))).unsqueeze(0), \
                torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
               torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(anchor_box.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(scale.astype(np.float32)).unsqueeze(0), gt_r, gt_t, bb3d

    def __len__(self):
        return self.length

    def enlarged_2d_box(self, cloud, cam_r = None, cam_t = None, eval = False):
        rmin = 10000
        rmax = -10000
        cmin = 10000
        cmax = -10000
        if eval == False:
            cloud = (cloud - cam_t) @ cam_r
        for tg in cloud:
            p1 = int(tg[0] * -280. / tg[2] + 127.5)
            p0 = int(tg[1] * 280 / tg[2] + 127.5)
            if p0 < rmin:
                rmin = p0
            if p0 > rmax:
                rmax = p0
            if p1 < cmin:
                cmin = p1
            if p1 > cmax:
                cmax = p1
        rmax += 1
        cmax += 1
        if rmin < 0:
            rmin = 0
        if cmin < 0:
            cmin = 0
        if rmax >= 256:
            rmax = 255
        if cmax >= 256:
            cmax = 255
        # print(rmin, rmax, cmin, cmax)
        r_b = rmax - rmin
        for tt in range(len(self.border_list)-1):
            if r_b > self.border_list[tt] and r_b < self.border_list[tt + 1]:
                r_b = self.border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(self.border_list)-1):
            if c_b > self.border_list[tt] and c_b < self.border_list[tt + 1]:
                c_b = self.border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)

        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > 256:
            delt = rmax - 256
            rmax = 256
            rmin -= delt
        if cmax > 256:
            delt = cmax - 256
            cmax = 256
            cmin -= delt

        # if rmin < 0:
        #     rmin = 0
        # if cmin < 0:
        #     cmin = 0

        if eval == True:
            return rmin, rmax, cmin, cmax
        # print(rmin, rmax, cmin, cmax)
        if ((rmax - rmin) in self.border_list) and ((cmax - cmin) in self.border_list):
            return rmin, rmax, cmin, cmax

        else:
            return 0

    def search_fit(self, points):
        min_x = min(points[:, 0])
        max_x = max(points[:, 0])
        min_y = min(points[:, 1])
        max_y = max(points[:, 1])
        min_z = min(points[:, 2])
        max_z = max(points[:, 2])

        return [min_x, max_x, min_y, max_y, min_z, max_z]

    def enlarge_bbox(self, target):

        limit = np.array(self.search_fit(target))
        longest = max(limit[1]-limit[0], limit[3]-limit[2], limit[5]-limit[4])
        longest = longest * self.e_scale
        scale1 = longest / (limit[1]-limit[0])
        scale2 = longest / (limit[3]-limit[2])
        scale3 = longest / (limit[5]-limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target


    def get_frame(self, index, video_path, in_cate, eval = False, current_r = None, current_t = None, full_img = False):

        # bb3d = sample['instances']['bboxes_3d'][in_cate][index]
        bb3d = np.load(os.path.join(video_path, 'bboxes_3d.npy'))[in_cate][index]
        r, t = self.get_pose(in_cate,  index, video_path)
        if self.eval == True:
            bb3d = self.enlarge_bbox((copy.deepcopy(bb3d) - t) @ r) # Object space
        else:
            bb3d =(copy.deepcopy(bb3d) - t) @ r
        ev = False
        if eval == True:
            bb3d = bb3d @ current_r.T + current_t # camera space
            ev = True
            c_r = None
            c_t = None
        else:
            bb3d = bb3d @ r.T + t # world space
            c_r = np.load(os.path.join(video_path, 'cam_r_' + str(index) + '.npy'))
            c_t = np.load(os.path.join(video_path, 'cam_t_' + str(index) + '.npy'))
        # Testing the 3d bounding box
        # fig, ax = plt.subplots(1)
        # ax.imshow(cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png')))
        # cloud = (bb3d - c_t) @ c_r # Cam space
        # for tg in cloud:
        #     p1 = int(tg[0] * -280. / tg[2] + 127.5)
        #     p0 = int(tg[1] * 280 / tg[2] + 127.5)
        #     plt.scatter(p1, p0)
        # plt.show()


        miny, maxy, minx, maxx  = self.enlarged_2d_box(bb3d, c_r, c_t, eval = ev)

        minv, maxv = np.load(os.path.join(video_path, 'depth_range' + '.npy'))
        depth = cv.imread(os.path.join(video_path, 'depth_' +str(index)+ '.png'))[:,:, 0] * 255. / 65535. * (maxv - minv) + minv

        if eval == True:
            bb3d = (bb3d - current_t) @ current_r # Object space
        # else:
        #     bb3d = (bb3d - t) @ r # Object space
        '''
        Depth test
        '''
        # # print(maxv, minv)
        # # print((bb3d - c_t) @ c_r)  # camera space
        # fig, ax = plt.subplots(1)
        # a = np.round(np.sqrt((cv.imread(os.path.join(video_path, 'depth_' +str(index)+ '.png'))[:,:, 0][:, :, np.newaxis] / 65535).clip(0, 1.)) * 255).astype(np.uint8)
        # ax.imshow(a)
        #
        # # ax.imshow(cv.imread(os.path.join(video_path, 'depth_' +str(index)+ '.png'))[:,:, 0][:, :, np.newaxis]/ 256 * (maxv - minv) + minv)
        # ax.imshow()
        # plt.show()

        '''
        Testing plot
        '''
        # fig, ax = plt.subplots(1)
        #
        # ax.imshow(cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png')))
        #
        # # ax.imshow(depth[:, np.newaxis][miny: maxy, minx: maxx])
        # rect = matplotlib.patches.Rectangle([minx, miny], maxx-minx, maxy-miny, linewidth =1, edgecolor = 'red', facecolor = 'none')
        # ax.add_patch(rect)
        # plt.show()
        if full_img == True:
            return bb3d, [np.transpose(
                cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png'))[miny: maxy, minx: maxx] / 255.,(2, 0, 1)), cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png')) ], depth[miny: maxy, minx: maxx], miny, maxy, minx, maxx
        if self.debug == True:
            return bb3d, np.transpose(
                cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png'))/ 255.,
                (2, 0, 1)), depth[miny: maxy, minx: maxx], miny, maxy, minx, maxx

        return bb3d , np.transpose(cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png'))[miny: maxy, minx: maxx] / 255., (2, 0, 1)), depth[miny: maxy, minx: maxx], miny, maxy, minx, maxx

    def get_pose(self, cate, frame_id, video_path):
        ins_r = np.load(os.path.join(video_path, 'instances_r' + '.npy'))
        ins_t = np.load(os.path.join(video_path, 'instances_t' + '.npy'))
        qa = np.quaternion(ins_r[cate][frame_id][0], ins_r[cate][frame_id][1], ins_r[cate][frame_id][2], ins_r[cate][frame_id][3])
        t = ins_t[cate][frame_id]
        r = quaternion.as_rotation_matrix(qa)
        return r, t

    def get_cloud(self, depth, miny, maxy, minx, maxx, video_path, index, limit, eval = False, current_r = None, current_t = None, cate_in = None, random_r = None, random_t = None, h = False):
        np.set_printoptions(threshold=np.inf)

        if self.mask == True:
            mask = np.load(os.path.join(video_path, 'mask.npy'))
            mask = (mask[index] == cate_in + 1)
            mask = mask[miny:maxy, minx:maxx].flatten()
            choose = (mask == True).nonzero()[0]

            if len(choose) == 0:
                return 0
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        else:
            choose = (depth.flatten() > -1000.).nonzero()[0]



        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)


        xmap_masked = self.xmap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / -1.
        # print(self.intrinsics[0][2], self.intrinsics[1][2], self.intrinsics[0][0], self.intrinsics[1][1])
        pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
        pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
        cloud = np.concatenate((-1. * pt0, pt1, pt2), axis=1)

        cloud = (cloud - current_t )@ current_r # object space
        if eval == False and self.debug == False:
            cloud  = cloud @ random_r.T + random_t # Do the random rotation and translation

        if self.mask == True:
            return cloud, choose
        else:
            choose_temp = (cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]) * (cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]) * (cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5])

            #     choose = (mask * choose_temp).nonzero()[0]
            # else:
            #     # print((depth.flatten() != 0.0).shape, choose_temp.shape)
            choose = ((depth.flatten() != 0.0) * choose_temp).nonzero()[0]

            if len(choose) == 0:
                return 0
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
            depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
            pt2 = depth_masked / -1.
            pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
            pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
            cloud = np.concatenate((-pt0, pt1, pt2), axis=1)
            cloud = (cloud - current_t )@ current_r # object space
            if eval == False and self.debug == False:
                cloud  = cloud @ random_r.T + random_t # Do the random rotation and translation


            return cloud , choose


    def points_vis(self, points, img, r, t, miny, maxy, minx, maxx, bb3d):
        # points: (1, 500, 3) in obj space
        limit = self.search_fit(bb3d)
        choose_temp = (points[:, 0] >= limit[0]) * (points[:, 0] <= limit[1]) * (points[:, 1] >= limit[2]) * (
                    points[:, 1] <= limit[3]) * (points[:, 2] >= limit[4]) * (points[:, 2] <= limit[5])
        # print('------------------------------------------------')
        # print(choose_temp)
        for i in points:
            if i[0] >= limit[0] and i[0] <= limit[1] and i[1] >= limit[2] and i[1] <= limit[3] and i[2] >= limit[4] and i[2] <= limit[5]:
                pass
            else:
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break


        points = points @ r.T + t # camera space
        bb3d = bb3d @ r.T + t


        m_proj = np.array([[-280., 0., 127.5], [0., 280., 127.5], [0., 0., 1.]])
        bb3d = (m_proj @ bb3d.transpose()).transpose()
        cam_bb3d = (m_proj @ points.transpose()).transpose()
        bb3d = ((1 / bb3d[:, 2]) * bb3d.transpose()).transpose()
        cam_bb3d = ((1 / cam_bb3d[:, 2]) * cam_bb3d.transpose()).transpose()
        cam_bb3d = cam_bb3d.astype(np.int)
        bb3d = bb3d.astype(np.int)
        points = cam_bb3d[:, 0:2]
        bb3d = bb3d[:, 0:2]
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for p in points:
            plt.scatter(p[0], p[1], s = 5)
        for b in bb3d:
            plt.scatter(b[0], b[1], s = 10, color = 'red')
        rect = matplotlib.patches.Rectangle([minx, miny], maxx - minx, maxy - miny, linewidth=1, edgecolor='red',
                                                facecolor='none')
        ax.add_patch(rect)
        plt.show()




    def get_anchor_box(self, ori_bbox):

        bbox = ori_bbox
        limit = np.array(self.search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale
    def times_scale(self, scale, pts):

        pts[:, 0] = pts[:, 0] * scale[0]
        pts[:, 1] = pts[:, 1] * scale[1]
        pts[:, 2] = pts[:, 2] * scale[2]

        return pts
    def divide_scale(self, scale, pts):

        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts
    def change_to_scale(self, scale, cloud_fr, cloud_to = None, eval = False):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        if eval == False:
            cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to
    def form_his(self, points_fr, points_to, rt, scale, r1, t1, r2, t2, cr_fr, ct_fr, cr_to, ct_to):
        relative_pose_fr = []
        relative_pose_to = []

        relative_cur_fr = [( rt[self.ms - 1][0][0] @ cr_fr.T).T, ct_fr - rt[self.ms - 1][0][1] @ rt[self.ms - 1][0][0] @ cr_fr.T]
        relative_cur_to = [( rt[self.ms - 1][1][0] @ cr_to.T).T, ct_to - rt[self.ms - 1][1][1] @ rt[self.ms - 1][1][0] @ cr_to.T]

        for i in range(self.ms -1):
            relative_pose_fr.append([(rt[i][0][0] @ rt[i + 1][0][0].T).T, rt[i+1][0][1] - rt[i][0][1] @ rt[i+1][0][0].T])
            relative_pose_to.append([(rt[i][1][0] @ rt[i + 1][1][0].T).T, rt[i + 1][1][1] - rt[i][1][1] @ rt[i + 1][1][0].T])

        for i in range(self.ms):
            ind = self.ms - i - 1
            points_fr[ind] = (points_fr[ind] - rt[ind][0][1]) @ rt[ind][0][0]  # Object space
            points_to[ind] = (points_to[ind] - rt[ind][1][1]) @ rt[ind][1][0]  # Object space

            if ind == self.ms - 1:
                # points_fr[ind] = (points_fr[ind] - rt[ind][0][1]) @ rt[ind][0][0] # Object space
                points_fr[ind] = points_fr[ind] @ r1.T @ relative_cur_fr[0] + (t1 - relative_cur_fr[1]) @ relative_cur_fr[0]
                # points_to[ind] = (points_to[ind] - rt[ind][1][1]) @ rt[ind][1][0]  # Object space
                points_to[ind] = points_to[ind] @ r2.T @ relative_cur_to[0] + (t2 - relative_cur_to[1] ) @ relative_cur_to[0]

                points_fr[ind] = torch.from_numpy(self.divide_scale(scale, (points_fr[ind]/self.dis_scale)).astype(np.float32))
                points_to[ind] = torch.from_numpy(self.divide_scale(scale, (points_to[ind]/self.dis_scale)).astype(np.float32))

                previous_r_fr = r1.T @ relative_cur_fr[0]
                previous_t_fr =  (t1 - relative_cur_fr[1]) @ relative_cur_fr[0]

                previous_r_to = r2.T @ relative_cur_to[0]
                previous_t_to = (t2 - relative_cur_to[1] ) @ relative_cur_to[0]


            else:

                points_fr[ind] = points_fr[ind] @ previous_r_fr @ relative_pose_fr[ind][0] + (previous_t_fr - relative_pose_fr[ind][1]) @ relative_pose_fr[ind][0]
                points_to[ind] = points_to[ind] @ previous_r_to @ relative_pose_to[ind][0] + (previous_t_to - relative_pose_to[ind][1]) @ relative_pose_to[ind][0]

                previous_r_fr = previous_r_fr @ relative_pose_fr[ind][0]
                previous_t_fr = (previous_t_fr - relative_pose_fr[ind][1]) @ relative_pose_fr[ind][0]

                previous_r_to = previous_r_to @ relative_pose_to[ind][0]
                previous_t_to = (previous_t_to - relative_pose_to[ind][1]) @ relative_pose_to[ind][0]

                points_fr[ind] = torch.from_numpy(self.divide_scale(scale, (points_fr[ind]/self.dis_scale)).astype(np.float32))
                points_to[ind] = torch.from_numpy(self.divide_scale(scale, (points_to[ind]/self.dis_scale)).astype(np.float32))

        return points_fr, points_to

    def __getitem__(self, index):

        while(True):
            try:
                choose_video = random.sample(range(self.video_num), 1)[0]
                video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(choose_video))
                bbox_frames = np.load(os.path.join(video_path, 'bbox_frames_n.npy'), allow_pickle = True)
                centers = np.load(os.path.join(video_path, 'centers.npy'))
                category = np.load(os.path.join(video_path, 'category.npy'))

                if self.ms != 0:
                    fr_his_fr = []
                    choose_his_fr = []
                    cloud_his_fr = []

                    fr_his_to = []
                    choose_his_to = []
                    cloud_his_to = []



                if self.cate not in category:
                    continue

                in_cate = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]

                while True:

                    # try:

                    if len(bbox_frames[in_cate]) < self.ms + 2:
                        sys.exit()

                    choose_frame = random.sample(list(bbox_frames[in_cate])[self.ms:], 2)
                    if choose_frame[0] == choose_frame[1]:

                        continue
                    # if choose_frame[0] >= choose_frame[1]:
                    #     continue
                    # if choose_frame[0] not in bbox_frames[in_cate] or choose_frame[1] not in bbox_frames[in_cate]:
                    #     continue

                    # if self.ms > 0  and list(bbox_frames[in_cate].numpy()).index(choose_frame[0]) < self.ms :
                    #     continue
                    # if self.mode == 'train':
                    if centers[in_cate][choose_frame[0]][0] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[0]][1] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[0]][1] * 256. >= 255 - self.padding or centers[in_cate][choose_frame[0]][0] * 256. >= 255 - self.padding:

                        sys.exit()
                    if centers[in_cate][choose_frame[1]][0] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[1]][1] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[1]][1] * 256. >= 255 - self.padding or centers[in_cate][choose_frame[1]][0] * 256. >= 255 - self.padding:

                        sys.exit()


                    '''
                    Obtaining current and next frames index, video_path, in_cate
                    '''
                    # bb3d: world space, frame: cropped rgb(3, h, w), depth: cropped depth(h, w)
                    fr_bb3d,  fr_frame, fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx = self.get_frame(choose_frame[0], video_path, in_cate)
                    to_bb3d,  to_frame, to_depth, to_miny, to_maxy, to_minx, to_maxx = self.get_frame(choose_frame[1], video_path, in_cate)
                    # Object pose
                    fr_r, fr_t = self.get_pose(in_cate, choose_frame[0], video_path)
                    to_r, to_t = self.get_pose(in_cate, choose_frame[1], video_path)

                    cr_fr = fr_r
                    ct_fr = fr_t
                    cr_to = to_r
                    ct_to = to_t

                    # Camera pose
                    fr_r_c, fr_t_c = np.load(os.path.join(video_path, 'cam_r_' + str(choose_frame[0]) + '.npy')), np.load(os.path.join(video_path, 'cam_t_' + str(choose_frame[0]) + '.npy'))
                    to_r_c, to_t_c = np.load(os.path.join(video_path, 'cam_r_' + str(choose_frame[1]) + '.npy')), np.load(os.path.join(video_path, 'cam_t_' + str(choose_frame[1]) + '.npy'))

                    # fr_r, fr_t = (fr_r.T @ fr_r_c).T, fr_r_c.T @ (fr_t - fr_t_c)
                    # to_r, to_t = (to_r.T @ to_r_c).T, to_r_c.T @ (to_t - to_t_c)

                    delta = math.pi / 10.
                    noise_trans = 0.05
                    r1 = euler_matrix(random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta))[:3, :3] # Use random transformation for training
                    t1 = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 10.

                    r2 = euler_matrix(random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta))[:3, :3] # Use random transformation for training
                    t2 = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 10.

                    # bb3d transformed from world space to obj space
                    fr_bb3d = (fr_bb3d - fr_t) @ fr_r
                    to_bb3d = (to_bb3d - to_t) @ to_r

                    fr_obj_r, fr_obj_t = (fr_r.T @ fr_r_c).T, fr_r_c.T @ (fr_t - fr_t_c) # Transformation from object to camera space
                    to_obj_r, to_obj_t = (to_r.T @ to_r_c).T, to_r_c.T @ (to_t - to_t_c)

                    limit_fr = self.search_fit(fr_bb3d) # The limit of 3d bounding box under obj space
                    limit_to = self.search_fit(to_bb3d)

                    if self.debug != True:
                        fr_bb3d /= self.dis_scale # Scale the 3d bounding box in obj space
                    anchor_box, scale = self.get_anchor_box(fr_bb3d) # The bounding box we use the obtain the anchor box is the one after scaling
                    if np.max(abs(anchor_box)) > 1.0:

                        sys.exit()

                    if self.mask == True:
                        c_i = in_cate
                    else:
                        c_i = None
                    # The points will be in obj space
                    fr_cloud, fr_choose = self.get_cloud(fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx, video_path, choose_frame[0], limit_fr, current_r= fr_obj_r, current_t = fr_obj_t, cate_in = c_i, random_r = r1, random_t = t1) # Return the cloud under obj space
                    to_cloud, to_choose = self.get_cloud(to_depth, to_miny, to_maxy, to_minx, to_maxx, video_path, choose_frame[1], limit_to, current_r=to_obj_r, current_t = to_obj_t, cate_in = c_i, random_r = r2, random_t = t2)



                    over_fr = (fr_cloud - t1) @ r1 # Negate the random transform
                    over_fr  = over_fr @ fr_obj_r.T + fr_obj_t # camera space


                    # fr_obj_r, fr_obj_t = (fr_r.T @ fr_r_c).T, fr_r_c.T @ (fr_t - fr_t_c)
                    # to_obj_r, to_obj_t = (to_r.T @ to_r_c).T, to_r_c.T @ (to_t - to_t_c)
                    #
                    # fr_cloud, to_cloud = (fr_cloud - fr_obj_t) @ fr_obj_r, (to_cloud - to_obj_t) @ to_obj_r # obj space point cloud

                    fr_r, fr_t = r1, t1 # Assign the random pose change to the valuable
                    to_r, to_t = r2, t2

                    fr_t /= self.dis_scale # Scale the two random translations.
                    to_t /= self.dis_scale
                    if self.debug != True:
                        fr_cloud /= self.dis_scale
                        to_cloud /= self.dis_scale # Scale the points finally.
                        fr_cloud, to_cloud = self.change_to_scale(scale, fr_cloud, to_cloud) # Scale the points again according to the scale factor from anchor box.
                    '''
                    Obtaining historical frames (Remember to add limit)
                    '''

                    # worlds_tr = []

                    for m in range(self.ms):
                        # his_index = bbox_frames[list(bbox_frames[in_cate].numpy()).index(choose_frame[0]) - self.ms - m]

                        his_index_fr = bbox_frames[in_cate][bbox_frames[in_cate].index(choose_frame[0]) - self.ms + m]
                        his_index_to = bbox_frames[in_cate][bbox_frames[in_cate].index(choose_frame[1]) - self.ms + m]

                        bb3_fr, his_frame_fr, his_depth_fr, his_miny_fr, his_maxy_fr, his_minx_fr, his_maxx_fr = self.get_frame(
                            his_index_fr, video_path, in_cate)

                        bb3_to, his_frame_to, his_depth_to, his_miny_to, his_maxy_to, his_minx_to, his_maxx_to = self.get_frame(
                            his_index_to, video_path, in_cate)
                        # Use obj space rather than world space
                        fr_r_o, fr_t_o = self.get_pose(in_cate, his_index_fr, video_path)
                        to_r_o, to_t_o = self.get_pose(in_cate, his_index_to, video_path)

                        fr_r_c, fr_t_c = np.load(
                            os.path.join(video_path, 'cam_r_' + str(his_index_fr) + '.npy')), np.load(
                            os.path.join(video_path, 'cam_t_' + str(his_index_fr) + '.npy'))
                        to_r_c, to_t_c = np.load(
                            os.path.join(video_path, 'cam_r_' + str(his_index_to) + '.npy')), np.load(
                            os.path.join(video_path, 'cam_t_' + str(his_index_to) + '.npy'))

                        fr_cr, fr_ct = (fr_r_o.T @ fr_r_c).T, fr_r_c.T @ (
                                    fr_t_o - fr_t_c)  # Transformation from object to camera space
                        to_cr, to_ct = (to_r_o.T @ to_r_c).T, to_r_c.T @ (to_t_o - to_t_c)

                        bb3_fr, bb3_to =(bb3_fr - fr_t_o) @ fr_r_o, (bb3_to - to_t_o) @ to_r_o # Obj space

                        his_cloud_fr, his_choose_fr = self.get_cloud(his_depth_fr, his_miny_fr, his_maxy_fr,
                                                                     his_minx_fr, his_maxx_fr, video_path, his_index_fr,
                                                                     limit=self.search_fit(bb3_fr), current_r=fr_cr, current_t=fr_ct, cate_in = c_i, eval = True)
                        his_cloud_to, his_choose_to = self.get_cloud(his_depth_to, his_miny_to, his_maxy_to,
                                                                     his_minx_to, his_maxx_to, video_path, his_index_to,
                                                                     limit=self.search_fit(bb3_to), current_r=to_cr, current_t=to_ct, cate_in = c_i, eval = True)

                        his_cloud_to, his_cloud_fr = his_cloud_to @ to_cr.T + to_ct, his_cloud_fr @ fr_cr.T + fr_ct # Cam space


                        # his_cloud_to, his_cloud_fr = his_cloud_to @ to_r_o.T + to_t_o, his_cloud_fr @ fr_r_o.T + fr_t_o # world space
                        # worlds_tr.append([[fr_r_o, fr_t_o], [to_r_o, to_t_o]])
                        # fr_mean = np.mean(his_cloud_fr, axis = 0, keepdims = True)
                        # to_mean = np.mean(his_cloud_to, axis = 0, keepdims = True)
                        #
                        # his_cloud_fr -= fr_mean
                        # his_cloud_to -= to_mean

                        # his_cloud_fr /= self.dis_scale
                        # his_cloud_to /= self.dis_scale

                        # his_cloud_fr = self.divide_scale(scale, his_cloud_fr)
                        # his_cloud_to = self.divide_scale(scale, his_cloud_to)
                        #fr_his_fr.append(his_frame_fr)
                        fr_his_fr.append(self.norm(torch.from_numpy(his_frame_fr.astype(np.float32))))
                        choose_his_fr.append(torch.LongTensor(his_choose_fr.astype(np.float32)))
                        cloud_his_fr.append(his_cloud_fr)
                        #cloud_his_fr.append(torch.from_numpy(his_cloud_fr.astype(np.float32)))

                        #
                        # else:
                        #fr_his_to.append(his_frame_to)
                        fr_his_to.append(self.norm(torch.from_numpy(his_frame_to.astype(np.float32))))
                        choose_his_to.append(torch.LongTensor(his_choose_to.astype(np.float32)))
                        cloud_his_to.append(his_cloud_to)
                        # cloud_his_to.append(torch.from_numpy(his_cloud_to.astype(np.float32)))


                    if self.ms != 0:

                        if self.opt.overlap_seq == True:
                            fr_his_to = copy.deepcopy(fr_his_fr)
                            choose_his_to = copy.deepcopy(choose_his_fr)
                            cloud_his_to = copy.deepcopy(cloud_his_fr)

                            fr_his_to.pop(0)
                            choose_his_to.pop(0)
                            cloud_his_to.pop(0)
                            fr_his_to.append(self.norm(torch.from_numpy(fr_frame.astype(np.float32))))
                            choose_his_to.append(torch.LongTensor(fr_choose.astype(np.int32)))
                            cloud_his_to.append(over_fr)

                        for mm in range(self.ms):
                            cloud_his_fr[mm] = (cloud_his_fr[mm] - fr_obj_t) @  fr_obj_r # Transform back to obj space using the current transformation
                            cloud_his_fr[mm] = torch.from_numpy(self.divide_scale(scale, (cloud_his_fr[mm] @ fr_r.T + fr_t * self.dis_scale )/ self.dis_scale).astype(np.float32)) # Do the same random transformation

                            cloud_his_to[mm] = (cloud_his_to[mm] - to_obj_t) @  to_obj_r
                            cloud_his_to[mm] = torch.from_numpy(self.divide_scale(scale, (cloud_his_to[mm] @ to_r.T + to_t * self.dis_scale )/ self.dis_scale).astype(np.float32))
                        # cloud_his_fr, cloud_his_to = self.form_his(cloud_his_fr, cloud_his_to, worlds_tr, scale, fr_r, fr_t * self.dis_scale,  to_r, to_t * self.dis_scale, cr_fr, ct_fr, cr_to, ct_to)

                    break
            except:
                continue
            break

        if self.ms != 0:
            return  self.norm(torch.from_numpy(fr_frame.astype(np.float32))), torch.from_numpy(fr_r.astype(np.float32)),\
                             torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)),\
                             torch.LongTensor(fr_choose.astype(np.int32)),\
                              self.norm(torch.from_numpy(to_frame.astype(np.float32))),\
                                       torch.from_numpy(to_r.astype(np.float32)),\
                                       torch.from_numpy(to_t.astype(np.float32)),\
                                       torch.from_numpy(to_cloud.astype(np.float32)),\
                                       torch.LongTensor(to_choose.astype(np.int32)), torch.from_numpy(anchor_box.astype(np.float32)), torch.from_numpy(scale.astype(np.float32)), fr_his_fr, choose_his_fr, cloud_his_fr,\
                                        fr_his_to, choose_his_to, cloud_his_to
        else:
            if self.debug == True:
                return torch.from_numpy(fr_frame.astype(np.float32)), torch.from_numpy(
                    fr_r.astype(np.float32)), \
                       torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)), \
                       torch.LongTensor(fr_choose.astype(np.int32)), \
                       self.norm(torch.from_numpy(to_frame.astype(np.float32))), \
                       torch.from_numpy(to_r.astype(np.float32)), \
                       torch.from_numpy(to_t.astype(np.float32)), \
                       torch.from_numpy(to_cloud.astype(np.float32)), \
                       torch.LongTensor(to_choose.astype(np.int32)), torch.from_numpy(
                    anchor_box.astype(np.float32)), torch.from_numpy(scale.astype(np.float32)), torch.from_numpy(fr_obj_r.astype(np.float32)), torch.from_numpy(fr_obj_t.astype(np.float32)), fr_miny, fr_maxy, fr_minx, fr_maxx, fr_bb3d

            return  self.norm(torch.from_numpy(fr_frame.astype(np.float32))), torch.from_numpy(fr_r.astype(np.float32)),\
                             torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)),\
                             torch.LongTensor(fr_choose.astype(np.int32)),\
                              self.norm(torch.from_numpy(to_frame.astype(np.float32))),\
                                       torch.from_numpy(to_r.astype(np.float32)),\
                                       torch.from_numpy(to_t.astype(np.float32)),\
                                       torch.from_numpy(to_cloud.astype(np.float32)),\
                                       torch.LongTensor(to_choose.astype(np.int32)), torch.from_numpy(anchor_box.astype(np.float32)), torch.from_numpy(scale.astype(np.float32))


