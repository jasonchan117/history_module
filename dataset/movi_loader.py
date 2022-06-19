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
import random
import torch
import matplotlib.patches
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML as html_print
import matplotlib.pyplot as plt
import quaternion

class Dataset(data.Dataset):
    def __init__(self, length, cate, num_pt = 1000, ms = 5):
        self.ms = ms
        self.num_pt = num_pt
        self.ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds",  shuffle_files = True)
        self.length = length
        self.cate = cate
        self.intrinsics = np.array([[280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        print(len(self.ds["train"]))
        print(len(self.ds['test']))
        self.ds = iter(tfds.as_numpy(self.ds["train"]))
        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __len__(self):
        return self.length

    def enlarged_2d_box(self, padding, minx, maxx, miny, maxy, resolution):
        minx = max(0, int(minx.numpy()) - padding) if type(minx) != int else max(0, minx - padding)
        maxx = min(resolution[0] - 1, int(maxx.numpy()) + padding) if type(maxx) != int else max(0, maxx + padding)
        miny = max(0, int(miny.numpy()) - padding) if type(miny) != int else max(0, miny - padding)
        maxy = min(resolution[1]- 1, int(maxy.numpy()) + padding) if type(maxy) != int else max(0, maxy + padding)
        return minx, maxx, miny, maxy

    def get_frame(self, index, sample, bboxes, bbox_frames, in_cate, resolution):
        idx = np.nonzero(bbox_frames[in_cate] == index)[0][0]
        miny, minx, maxy, maxx = bboxes[in_cate][idx]
        miny = max(1, miny * resolution[0])
        minx = max(1, minx * resolution[1])
        maxy = min(resolution[0] - 1, maxy * resolution[0])
        maxx = min(resolution[1] - 1, maxx * resolution[1])
        minx, maxx, miny, maxy = self.enlarged_2d_box(30, minx, maxx, miny, maxy, resolution)
        minv, maxv = sample['metadata']['depth_range']
        depth = sample["depth"] / 65535 * (maxv - minv) + minv


        return np.transpose(sample['video'][index][miny: maxy, minx: maxx] / 255., (2, 0, 1)), depth[index][miny: maxy, minx: maxx], miny, maxy, minx, maxx

    def get_pose(self, cate, frame_id, sample):
        qa = np.quaternion(sample['instances']['quaternions'][cate][frame_id][0], sample['instances']['quaternions'][cate][frame_id][1], sample['instances']['quaternions'][cate][frame_id][2], sample['instances']['quaternions'][cate][frame_id][3])
        t = sample['instances']['positions'][cate][frame_id]
        r = quaternion.as_rotation_matrix(qa)
        return r, t
    def get_cloud(self, depth, miny, maxy, minx, maxx, camera_r, camera_t):
        choose = (depth.flatten() > -1000.).nonzero()[0]
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
        pt2 = depth
        pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
        pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
        cloud = cloud @ camera_r + camera_t
        return cloud / 100. , choose
    def __getitem__(self, index):

        while(True):
            sample = next(self.ds)
            bboxes = sample["instances"]["bboxes"]
            bbox_frames = sample["instances"]["bbox_frames"]
            resolution = sample["video"].shape[-3:-1]
            category = sample['instances']['category']
            flag = 0
            if self.ms != 0:
                fr_his = []
                choose_his = []
                cloud_his = []
                t_his = []

            for in_cate, ca in enumerate(category):

                if ca == self.cate:
                    flag = 1
                    while True:
                        choose_frame = random.sample(range(24), 2)
                        if choose_frame[0] >= choose_frame[1]:
                            continue
                        if choose_frame[0] not in bbox_frames[in_cate] or choose_frame[1] not in bbox_frames[in_cate]:
                            continue
                        if choose_frame[0] < self.ms:
                            continue
                        if self.ms > 0  and  [i for i in range(choose_frame[0] - self.ms , choose_frame[0])] not in bbox_frames[in_cate]:
                            continue


                        for m in range(self.ms):
                            his_frame, his_depth, his_miny, his_maxy, his_minx, his_maxx = self.get_frame(choose_frame[0] - (self.ms - m),
                                                                                                    sample, bboxes,
                                                                                                    bbox_frames,
                                                                                                    in_cate, resolution)
                            _, his_t = self.get_pose(in_cate, choose_frame[0] - (self.ms - m), sample)
                            his_cloud, his_choose = self.get_cloud(his_depth, his_miny, his_maxy, his_minx, his_maxx,
                                                  quaternion.as_rotation_matrix(
                                                      np.quaternion(sample['camera']['quaternions'][choose_frame[0] - (self.ms - m)][0],
                                                                    sample['camera']['quaternions'][choose_frame[0] - (self.ms - m)][1],
                                                                    sample['camera']['quaternions'][choose_frame[0] - (self.ms - m)][2],
                                                                    sample['camera']['quaternions'][choose_frame[0] - (self.ms - m)][3])),
                                                  sample['camera']['positions'][[choose_frame[0] - (self.ms - m)]])
                            fr_his.append(self.norm(torch.from_numpy(his_frame.astype(np.float32))))
                            choose_his.append(torch.LongTensor(his_choose.astype(np.float32)))
                            cloud_his.append(torch.from_numpy(his_cloud.astype(np.float32)))
                            t_his.append(torch.from_numpy(his_t.astype(np.float32)))
                        fr_frame, fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx = self.get_frame(choose_frame[0], sample, bboxes, bbox_frames, in_cate, resolution)
                        to_frame, to_depth, to_miny, to_maxy, to_minx, to_maxx = self.get_frame(choose_frame[1], sample, bboxes, bbox_frames, in_cate, resolution)
                        fr_r, fr_t = self.get_pose(in_cate, choose_frame[0], sample)
                        to_r, to_t = self.get_pose(in_cate, choose_frame[1], sample)
                        fr_cloud, fr_choose = self.get_cloud(fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx, quaternion.as_rotation_matrix(np.quaternion(sample['camera']['quaternions'][choose_frame[0]][0], sample['camera']['quaternions'][choose_frame[0]][1], sample['camera']['quaternions'][choose_frame[0]][2], sample['camera']['quaternions'][choose_frame[0]][3])), sample['camera']['positions'][[choose_frame[0]]])
                        to_cloud, to_choose = self.get_cloud(to_depth, to_miny, to_maxy, to_minx, to_maxx,
                                                  quaternion.as_rotation_matrix(
                                                      np.quaternion(sample['camera']['quaternions'][choose_frame[1]][0],
                                                                    sample['camera']['quaternions'][choose_frame[1]][1],
                                                                    sample['camera']['quaternions'][choose_frame[1]][2],
                                                                    sample['camera']['quaternions'][choose_frame[1]][3])),
                                                  sample['camera']['positions'][[choose_frame[1]]])
                        # np.set_printoptions(threshold = np.inf)


                        break
                    break

            if flag == 0:
                continue
            break

        if self.ms != 0:
            return self.norm(torch.from_numpy(fr_frame.astype(np.float32))), torch.from_numpy(fr_r.astype(np.float32)),\
                             torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)),\
                             torch.LongTensor(fr_choose.astype(np.int32)),\
                             self.norm(torch.from_numpy(to_frame.astype(np.float32))),\
                                       torch.from_numpy(to_r.astype(np.float32)),\
                                       torch.from_numpy(to_t.astype(np.float32)),\
                                       torch.from_numpy(to_cloud.astype(np.float32)),\
                                       torch.LongTensor(to_choose.astype(np.int32)), fr_his, choose_his, cloud_his,\
                                       t_his
        else:
            return self.norm(torch.from_numpy(fr_frame.astype(np.float32))), torch.from_numpy(fr_r.astype(np.float32)),\
                             torch.from_numpy(fr_t.astype(np.float32)), torch.from_numpy(fr_cloud.astype(np.float32)),\
                             torch.LongTensor(fr_choose.astype(np.int32)),\
                             self.norm(torch.from_numpy(to_frame.astype(np.float32))),\
                                       torch.from_numpy(to_r.astype(np.float32)),\
                                       torch.from_numpy(to_t.astype(np.float32)),\
                                       torch.from_numpy(to_cloud.astype(np.float32)),\
                                       torch.LongTensor(to_choose.astype(np.int32))

dataset = Dataset(5000, 16 ,ms = 0 )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for i, data in enumerate(dataloader, 0):
    fr_frame, fr_r, fr_t, fr_cloud, fr_choose, to_frame, to_r, to_t, to_cloud, to_choose  = data
    print(fr_frame.shape, fr_r.shape, fr_t.shape, fr_cloud.shape, fr_choose.shape)




