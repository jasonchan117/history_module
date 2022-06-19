import seaborn as sns
import matplotlib
import matplotlib.colors
import numpy as np
import mediapy as media
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import PIL
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
    def __init__(self, length, cate):

        self.ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds",  shuffle_files = True)
        self.length = length
        self.cate = cate
        self.intrinsics = np.array([[280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        print(len(self.ds["train"]))
        print(len(self.ds['test']))
        self.ds = iter(tfds.as_numpy(self.ds["train"]))
        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])

    def __len__(self):
        return self.length

    def enlarged_2d_box(self, padding, minx, maxx, miny, maxy, resolution):
        minx = max(0, minx - padding)
        maxx = min(resolution[0] - 1, maxx + padding)
        miny = max(0, miny - padding)
        maxy = min(resolution[1]- 1, maxy + padding)
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
        miny, minx, maxy, maxx = int(miny.numpy()), int(minx.numpy()), int(maxy.numpy()), int(maxx.numpy())
        return sample['video'][index][miny: maxy, minx: maxx] / 255., depth[index][miny: maxy, minx: maxx], miny, maxy, minx, maxx

    def get_pose(self, cate, frame_id, sample):
        qa = np.quaternion(sample['instances']['quaternion'][cate][frame_id][0], sample['instances']['quaternion'][cate][frame_id][1], sample['instances']['quaternion'][cate][frame_id][2], sample['instances']['quaternion'][cate][frame_id][3])
        t = sample['instances']['position'][cate][frame_id]
        r = qa.as_rotation_matrix(qa)
        return r, t
    def get_cloud(self, depth, miny, maxy, minx, maxx):
        choose = (depth.flatten() > -1000.).nonzero()[0]
        depth = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[miny:maxy, minx:maxx].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth
        pt0 = (ymap_masked - self.intrinsics[0][2]) * pt2 / self.intrinsics[0][0]
        pt1 = (xmap_masked - self.intrinsics[1][2]) * pt2 / self.intrinsics[1][1]
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

    def __getitem__(self, index):

        sample = next(self.ds)

        bboxes = sample["instances"]["bboxes"]
        bbox_frames = sample["instances"]["bbox_frames"]
        resolution = sample["video"].shape[-3:-1]
        bboxes_3d = sample['instances']['bboxes_3d']
        category = sample['instances']['category']

        for in_cate, ca in enumerate(category):
            if ca == self.cate:
                while True:
                    choose_frame = random.sample(range(24), 2)
                    if choose_frame[0] >= choose_frame[1]:
                        continue
                    if choose_frame[0] not in bbox_frames[in_cate] or choose_frame[1] not in bbox_frames[in_cate]:
                        continue

                    fr_frame, fr_depth, fr_miny, fr_maxy, fr_minx, fr_maxx = self.get_frame(choose_frame[0], sample, bboxes, bbox_frames, in_cate, resolution)
                    to_frame, to_depth, to_miny, to_maxy, to_minx, to_maxx = self.get_frame(choose_frame[1], sample, bboxes, bbox_frames, in_cate, resolution)
                    fr_r, fr_t = self.get_pose(in_cate, choose_frame[0], sample)
                    to_r, to_t = self.get_pose(in_cate, choose_frame[1], sample)

                    break

        # np.linalg.inv()
        # sys.exit()
        # bb_3d = sample['instances']['bboxes_3d'][0][10]
        #
        # r = R.from_quat(sample['instances']['quaternions'][0][10]).as_matrix()
        # t = sample['instances']['positions'][0][10]
        #
        #
        # c_q = np.quaternion(sample['camera']['quaternions'][10][0], sample['camera']['quaternions'][10][1], sample['camera']['quaternions'][10][2], sample['camera']['quaternions'][10][3])
        #
        # c_pos = sample['camera']['positions'][10]
        #
        #
        # bb_3d -= c_pos
        # for i in range(8):
        #     bb_3d[i] = quaternion.as_rotation_matrix(c_q).T @ bb_3d[i]
        # obj_cam = bb_3d
        # # obj_cam = quaternion.rotate_vectors(c_q.conjugate(), bb_3d - c_pos)
        # # obj_cam = quaternion.rotate_vectors(c_q, bb_3d ) + c_pos
        # # obj_cam = np.dot(np.linalg.inv(R.from_quat(sample['camera']['quaternions'][10]).as_matrix()) , )
        #
        # # for i in range(8):
        # #     obj_cam[i] = np.dot(R.from_quat( [0.7673993,  0.44150472, 0.23185928, 0.4030051 ]).as_matrix().T, obj_cam[i].transpose())
        # # obj_cam = quaternion.rotate_vector(c_q.conjugate(), bb_3d - sample['camera']['positions'][10])
        #
        # m_proj = np.array([[-280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        # print(m_proj)
        # obj_img = (m_proj @ obj_cam.transpose()).transpose()
        # obj_img = ((1/obj_img[:, 2]) * obj_img.transpose()).transpose()
        # print(obj_img)
        #
        #
        # camera_r = R.from_quat(sample['camera']['quaternions'][10]).as_matrix()
        # camera_t = sample['camera']['positions'][10]
        #
        # # scale = sample['instances']['scale'][0]
        #
        #
        # # img_bb_3d = np.matmul(bb_3d - t, np.linalg.inv(r))
        # img_bb_3d = np.matmul(bb_3d , camera_r) + camera_t
        #
        # for i in range(8):
        #     for j in range(3):
        #         img_bb_3d[i][j] /= img_bb_3d[i][2]
        #
        #
        #
        # miny, minx, maxy, maxx = bboxes[0][10]
        # miny = max(1, miny * resolution[0])
        # minx = max(1, minx * resolution[1])
        # maxy = min(resolution[0] - 1, maxy * resolution[0])
        # maxx = min(resolution[1] - 1, maxx * resolution[1])
        # for i in range(8):
        #     img_bb_3d[i][0] = (img_bb_3d[i][0] * focal_length * 256. / sample['camera']['sensor_width']) + 128
        #     img_bb_3d[i][1] = (img_bb_3d[i][1] * focal_length * 256. / sample['camera']['sensor_width']) + 128
        #
        # fig, ax = plt.subplots(1)
        # frame = sample['video'][10]
        # ax.imshow(frame)
        # rect = matplotlib.patches.Rectangle([minx, miny], maxx - minx, maxy - miny,
        #                                     linewidth=1, edgecolor='r',
        #                                     facecolor='none')
        # ax.add_patch(rect)
        # # ax.xaxis.set_ticks_position('top')
        # # ax.invert_yaxis()
        # # ax.invert_yaxis()
        # # img_bb_3d *= scale
        #
        # for i in range(8):
        #
        #     plt.scatter(obj_img[i][0], obj_img[i][1], edgecolors='blue')
        # plt.show()
        return depth

dataset = Dataset(5000, 6 )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for i, data in enumerate(dataloader, 0):
    depth  = data




