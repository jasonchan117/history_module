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
import quaternion
c = ["Action Figures", "Bag", "Board Games", "Bottles and Cans and Cups", "Camera", "Car Seat", "Consumer Goods", "Hat", "Headphones", "Keyboard", "Legos", "Media Cases", "Mouse", "None", "Shoe", "Stuffed Toys", "Toys"]


# with tf.device('/cpu:0'):
class Dataset(data.Dataset):
    def __init__(self, opt, mode = 'train', length = 5000, eval = False):
        self.ms = opt.memory_size
        self.padding = 10
        self.eval = eval
        self.num_pt = opt.num_pt
        self.opt = opt
        self.root = opt.dataset_root
        self.mode = mode
        self.video_num = len(os.listdir(os.path.join(opt.dataset_root, mode, c[opt.category])))
        self.length = length
        self.dis_scale = 10.
        self.cate = opt.category
        self.intrinsics = np.array([[280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        self.border_list = [-1, 80, 120, 160, 200, 240, 280]
        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        if self.eval == True:
            self.current_video_num = 0
            video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(self.current_video_num))
            current_object = np.load(os.path.join(video_path, 'bbox_frames_n.npy'), allow_pickle=True)
            video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(self.current_video_num))
            category = np.load(os.path.join(video_path, 'category.npy'))
            in_cate = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]

    def __len__(self):
        return self.length

    def enlarged_2d_box(self, cloud, cam_r, cam_t):
        rmin = 10000
        rmax = -10000
        cmin = 10000
        cmax = -10000
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

        if rmin < 0:
            rmin = 0
        if cmin < 0:
            cmin = 0

        # print(rmin, rmax, cmin, cmax)
        # if ((rmax - rmin) in self.border_list) and ((cmax - cmin) in self.border_list):
        return rmin, rmax, cmin, cmax

        # else:
        #     return 0

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
        longest = longest * 1.1
        scale1 = longest / (limit[1]-limit[0])
        scale2 = longest / (limit[3]-limit[2])
        scale3 = longest / (limit[5]-limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target


    def get_frame(self, index, video_path, in_cate):

        # bb3d = sample['instances']['bboxes_3d'][in_cate][index]
        bb3d = np.load(os.path.join(video_path, 'bboxes_3d.npy'))[in_cate][index]
        r, t = self.get_pose(in_cate,  index, video_path)
        bb3d = self.enlarge_bbox((copy.deepcopy(bb3d) - t) @ r) # Object space
        bb3d = bb3d @ r.T + t # world space
        c_r = np.load(os.path.join(video_path, 'cam_r_' + str(index) + '.npy'))
        c_t = np.load(os.path.join(video_path, 'cam_t_' + str(index) + '.npy'))
        miny, maxy, minx, maxx  = self.enlarged_2d_box(bb3d, c_r, c_t)

        minv, maxv = np.load(os.path.join(video_path, 'depth_range' + '.npy'))
        # depth = np.round(np.sqrt((cv.imread(os.path.join(video_path, 'depth_' +str(index)+ '.png'))[:,:, 0][:, :, np.newaxis] / 65535).clip(0, 1.)) * 255).astype(np.uint8)
        depth = cv.imread(os.path.join(video_path, 'depth_' +str(index)+ '.png'))[:,:, 0] * 256 / 65535. * (maxv - minv) + minv
        # depth = cv.imread(os.path.join(video_path, 'depth_' + str(index) + '.png'))[:, :, 0]
        # print(cv.imread(os.path.join(video_path, 'depth_' + str(index) + '.png'))[:, :, 0])
        #
        # print(cv.imread(os.path.join(video_path, 'depth_' + str(index) + '.png'))[:, :, 1])

        '''
        Depth test
        '''
        # print(maxv, minv)
        # print((bb3d - c_t) @ c_r)  # camera space
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
        # print('>>>', sample['video'][index].shape)
        # ax.imshow(sample['video'][index])
        # ax.imshow(sample['video'][index][miny: maxy, minx: maxx])
        # plt.show()
        return bb3d , np.transpose(cv.imread(os.path.join(video_path, 'rgb_' + str(index) + '.png'))[miny: maxy, minx: maxx] / 255., (2, 0, 1)), depth[miny: maxy, minx: maxx], miny, maxy, minx, maxx

    def get_pose(self, cate, frame_id, video_path):
        ins_r = np.load(os.path.join(video_path, 'instances_r' + '.npy'))
        ins_t = np.load(os.path.join(video_path, 'instances_t' + '.npy'))
        qa = np.quaternion(ins_r[cate][frame_id][0], ins_r[cate][frame_id][1], ins_r[cate][frame_id][2], ins_r[cate][frame_id][3])
        t = ins_t[cate][frame_id]
        r = quaternion.as_rotation_matrix(qa)
        return r, t

    def get_cloud(self, depth, miny, maxy, minx, maxx, video_path, index, limit):
        np.set_printoptions(threshold=np.inf)
        choose = (depth.flatten() > -1000.).nonzero()[0]

        camera_r = np.load(os.path.join(video_path, 'cam_r_' + str(index) + '.npy'))

        camera_t = np.load(os.path.join(video_path, 'cam_t_' + str(index) + '.npy'))
        if len(choose) == 0:
            return 0
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth = depth.flatten()[choose][:, np.newaxis].astype(np.float32)


        xmap_masked = self.xmap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth / -1.
        # print(self.intrinsics[0][2], self.intrinsics[1][2], self.intrinsics[0][0], self.intrinsics[1][1])
        pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
        pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
        cloud = np.concatenate((-1. * pt0, pt1, pt2), axis=1)

        cloud = cloud @ camera_r.T + camera_t # world space

        # print((cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]))
        # print('-----------------')
        # print((cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]))
        # print('-----------------')
        # print((cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5]))

        choose_temp = (cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]) * (cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]) * (cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5])
        # np.set_printoptions(threshold = np.inf)

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
        depth = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth / -1.
        pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
        pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
        cloud = np.concatenate((-pt0, pt1, pt2), axis=1)
        cloud = cloud @ camera_r.T + camera_t # world space
        return cloud , choose



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

    def divide_scale(self, scale, pts):

        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts
    def change_to_scale(self, scale, cloud_fr, cloud_to):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to
    def get_init_pose(self, video_num, in_cate):
        video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(video_num))
        bbox_frames = np.load(os.path.join(video_path, 'bbox_frames_n.npy'), allow_pickle=True)

    def __getitem__(self, index):

        while(True):
            try:
                choose_video = random.sample(range(self.video_num), 1)[0]
                video_path = os.path.join(self.opt.dataset_root, self.mode, c[self.opt.category], str(choose_video))
                bbox_frames = np.load(os.path.join(video_path, 'bbox_frames_n.npy'), allow_pickle = True)
                centers = np.load(os.path.join(video_path, 'centers.npy'))
                category = np.load(os.path.join(video_path, 'category.npy'))

                flag = 0
                if self.ms != 0:
                    fr_his = []
                    choose_his = []
                    cloud_his = []
                    t_his = []

                if self.cate not in category:
                    continue

                in_cate = random.sample(list(np.argwhere(category == self.cate)), 1)[0][0]

                while True:
                    # try:

                    if len(bbox_frames[in_cate]) < self.ms + 2:
                        flag = 1
                        break
                    choose_frame = random.sample(range(24), 2)
                    if choose_frame[0] >= choose_frame[1]:
                        continue
                    if choose_frame[0] not in bbox_frames[in_cate] or choose_frame[1] not in bbox_frames[in_cate]:
                        continue
                    if choose_frame[0] < self.ms:
                        continue
                    if self.ms > 0  and list(bbox_frames[in_cate].numpy()).index(choose_frame[0]) < self.ms :
                        continue

                    if centers[in_cate][choose_frame[0]][0] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[0]][1] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[0]][1] * 256. >= 255 - self.padding or centers[in_cate][choose_frame[0]][0] * 256. >= 255 - self.padding:

                        sys.exit()
                    if centers[in_cate][choose_frame[1]][0] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[1]][1] * 256. <= 1 + self.padding or centers[in_cate][choose_frame[1]][1] * 256. >= 255 - self.padding or centers[in_cate][choose_frame[1]][0] * 256. >= 255 - self.padding:

                        sys.exit()
                    '''
                    Obtaining historical frames (Remember to add limit)
                    '''
                    for m in range(self.ms):
                        his_index = bbox_frames[list(bbox_frames[in_cate].numpy()).index(choose_frame[0]) - self.ms - m]
                        _ ,  his_frame, his_depth, his_miny, his_maxy, his_minx, his_maxx = self.get_frame(his_index, video_path, in_cate)
                        _, his_t = self.get_pose(in_cate, his_index, video_path)
                        his_cloud, his_choose = self.get_cloud(his_depth, his_miny, his_maxy, his_minx, his_maxx, video_path, his_index)
                        fr_his.append(self.norm(torch.from_numpy(his_frame.astype(np.float32))))
                        choose_his.append(torch.LongTensor(his_choose.astype(np.float32)))
                        cloud_his.append(torch.from_numpy(his_cloud.astype(np.float32)))
                        t_his.append(torch.from_numpy(his_t.astype(np.float32)))

                    '''
                    Obtaining current and next frames index, video_path, in_cate
                    '''

                    fr_bb3d,  fr_frame, fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx = self.get_frame(choose_frame[0], video_path, in_cate)
                    to_bb3d,  to_frame, to_depth, to_miny, to_maxy, to_minx, to_maxx = self.get_frame(choose_frame[1], video_path, in_cate)

                    fr_r, fr_t = self.get_pose(in_cate, choose_frame[0], video_path)
                    to_r, to_t = self.get_pose(in_cate, choose_frame[1], video_path)


                    limit_fr = self.search_fit(fr_bb3d)
                    limit_to = self.search_fit(to_bb3d)
                    # fr_bb3d = (fr_bb3d - fr_t) @ fr_r # object space
                    #lim_fr = self.search_fit((fr_bb3d - fr_t) @ fr_r)
                    #lim_to = self.search_fit((to_bb3d - to_t) @ to_r)
                    fr_bb3d /= self.dis_scale
                    anchor_box, scale = self.get_anchor_box(fr_bb3d)

                    fr_t /= self.dis_scale
                    to_t /= self.dis_scale


                    fr_cloud, fr_choose = self.get_cloud(fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx, video_path, choose_frame[0], limit_fr)
                    to_cloud, to_choose = self.get_cloud(to_depth, to_miny, to_maxy, to_minx, to_maxx, video_path, choose_frame[1], limit_to)

                    fr_cloud /= self.dis_scale
                    to_cloud /= self.dis_scale

                    # np.set_printoptions(threshold = np.inf)
                    fr_cloud, to_cloud = self.change_to_scale(scale, fr_cloud, to_cloud)
                    # except:
                    #     # print('loader error')
                    #     continue
                    break

                if flag == 1:
                    continue
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
                                       torch.LongTensor(to_choose.astype(np.int32)), torch.from_numpy(anchor_box.astype(np.float32)), torch.from_numpy(scale.astype(np.float)), fr_his, choose_his, cloud_his,\
                                       t_his
        else:
            return  self.norm(torch.from_numpy(fr_frame.astype(np.float32))), torch.from_numpy(fr_r.astype(np.float32)),\
                             torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)),\
                             torch.LongTensor(fr_choose.astype(np.int32)),\
                              self.norm(torch.from_numpy(to_frame.astype(np.float32))),\
                                       torch.from_numpy(to_r.astype(np.float32)),\
                                       torch.from_numpy(to_t.astype(np.float32)),\
                                       torch.from_numpy(to_cloud.astype(np.float32)),\
                                       torch.LongTensor(to_choose.astype(np.int32)), torch.from_numpy(anchor_box.astype(np.float32)), torch.from_numpy(scale.astype(np.float32))


