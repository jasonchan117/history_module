
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

        self.ds, self.ds_info = tfds.load("movi_a", data_dir="gs://kubric-public/tfds", with_info=True, shuffle_files = True)
        self.length = length
        self.cate = cate
        self.ds = iter(tfds.as_numpy(self.ds["train"]))
        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])
    def __len__(self):
        return self.length

    def plot_bboxes(sample, palette=None, linewidth=1):
        resolution = sample["video"].shape[-3:-1]

        bboxes = sample["instances"]["bboxes"]
        bbox_frames = sample["instances"]["bbox_frames"]
        num_objects = bboxes.shape[0]
        if palette is None:
            palette = sns.color_palette('hls', num_objects)
        images = []
        for t, rgb in enumerate(sample["video"]):
            fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=132.5)
            ax.axis("off")
            ax.imshow(rgb)
            for k in range(num_objects):
                if t in bbox_frames[k]:
                    idx = np.nonzero(bbox_frames[k] == t)[0][0]

                    miny, minx, maxy, maxx = bboxes[k][idx]
                    miny = max(1, miny * resolution[0])
                    minx = max(1, minx * resolution[1])
                    maxy = min(resolution[0] - 1, maxy * resolution[0])
                    maxx = min(resolution[1] - 1, maxx * resolution[1])
                    rect = matplotlib.patches.Rectangle([minx, miny], maxx - minx, maxy - miny,
                                                        linewidth=linewidth, edgecolor=palette[k],
                                                        facecolor='none')
                    ax.add_patch(rect)

            for k in range(num_objects):
                x, y = sample["instances"]["image_positions"][k, t] * resolution
                if np.all(1 < y < resolution[0] - 1) and np.all(1 < x < resolution[1] - 1):
                    ax.scatter(x, y, marker="X", s=5, color=palette[k])
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, dpi=132.5)
            plt.close(fig)
            buf.seek(0)
            img = PIL.Image.open(buf)
            images.append(np.array(img)[..., :3])
        return images
    def get_frame(self, index, sample, bboxes, bbox_frames, in_cate, resolution):
        idx = np.nonzero(bbox_frames[in_cate] == index)[0][0]
        miny, minx, maxy, maxx = bboxes[in_cate][idx]
        miny = max(1, miny * resolution[0])
        minx = max(1, minx * resolution[1])
        maxy = min(resolution[0] - 1, maxy * resolution[0])
        maxx = min(resolution[1] - 1, maxx * resolution[1])
        minv, maxv = sample['metadata']['depth_range']
        depth = sample["depth"] / 65535 * (maxv - minv) + minv
        return sample['video'][index][minx: maxx, miny: maxy], depth[index][minx: maxx, miny: maxy]

    def get_cloud(self, ):
        pass
    def __getitem__(self, index):

        sample = next(self.ds)

        # category = sample['instances']['category']
        bboxes = sample["instances"]["bboxes"]
        bbox_frames = sample["instances"]["bbox_frames"]
        focal_length = sample['camera']['focal_length']
        resolution = sample["video"].shape[-3:-1]
        bboxes_3d = sample['instances']['bboxes_3d']
        # for in_cate, ca in enumerate(category):
        #     if ca == self.cate:
        #         while True:
        #             choose_frame = random.sample(range(24), 2)
        #             if choose_frame[0] >= choose_frame[1]:
        #                 continue
        #             if choose_frame[0] not in bbox_frames[in_cate] or choose_frame[1] not in bbox_frames[in_cate]:
        #                 continue
        #
        #             fr_frame, fr_depth = self.get_frame(choose_frame[0], sample, bboxes, bbox_frames, in_cate, resolution)
        #             to_frame, to_depth = self.get_frame(choose_frame[0], sample, bboxes, bbox_frames, in_cate, resolution)
        #             break
        #
        # np.linalg.inv()
        bb_3d = sample['instances']['bboxes_3d'][0][10]

        r = R.from_quat(sample['instances']['quaternions'][0][10]).as_matrix()
        t = sample['instances']['positions'][0][10]


        c_q = np.quaternion(sample['camera']['quaternions'][10][0], sample['camera']['quaternions'][10][1], sample['camera']['quaternions'][10][2], sample['camera']['quaternions'][10][3])

        c_pos = sample['camera']['positions'][10]

        obj_cam = quaternion.rotate_vectors(c_q.conjugate(), bb_3d - c_pos)
        # obj_cam = quaternion.rotate_vectors(c_q, bb_3d ) + c_pos
        # obj_cam = np.dot(np.linalg.inv(R.from_quat(sample['camera']['quaternions'][10]).as_matrix()) , )

        # for i in range(8):
        #     obj_cam[i] = np.dot(R.from_quat( [0.7673993,  0.44150472, 0.23185928, 0.4030051 ]).as_matrix().T, obj_cam[i].transpose())
        # obj_cam = quaternion.rotate_vector(c_q.conjugate(), bb_3d - sample['camera']['positions'][10])

        m_proj = np.array([[280., 0., 127.5],[0., 280., 127.5], [0.,0.,1.]])
        print(m_proj)
        obj_img = (m_proj @ obj_cam.transpose()).transpose()
        obj_img = ((1/obj_img[:, 2]) * obj_img.transpose()).transpose()
        print(obj_img)


        camera_r = R.from_quat(sample['camera']['quaternions'][10]).as_matrix()
        camera_t = sample['camera']['positions'][10]

        # scale = sample['instances']['scale'][0]


        # img_bb_3d = np.matmul(bb_3d - t, np.linalg.inv(r))
        img_bb_3d = np.matmul(bb_3d , camera_r) + camera_t

        for i in range(8):
            for j in range(3):
                img_bb_3d[i][j] /= img_bb_3d[i][2]



        miny, minx, maxy, maxx = bboxes[0][10]
        miny = max(1, miny * resolution[0])
        minx = max(1, minx * resolution[1])
        maxy = min(resolution[0] - 1, maxy * resolution[0])
        maxx = min(resolution[1] - 1, maxx * resolution[1])
        for i in range(8):
            img_bb_3d[i][0] = (img_bb_3d[i][0] * focal_length * 256. / sample['camera']['sensor_width']) + 128
            img_bb_3d[i][1] = (img_bb_3d[i][1] * focal_length * 256. / sample['camera']['sensor_width']) + 128

        fig, ax = plt.subplots(1)
        frame = sample['video'][10]
        ax.imshow(frame)
        rect = matplotlib.patches.Rectangle([minx, miny], maxx - minx, maxy - miny,
                                            linewidth=1, edgecolor='r',
                                            facecolor='none')
        ax.add_patch(rect)
        # ax.xaxis.set_ticks_position('top')
        # ax.invert_yaxis()
        # ax.invert_yaxis()
        # img_bb_3d *= scale

        for i in range(8):

            plt.scatter(obj_img[i][0], obj_img[i][1], edgecolors='blue')
        plt.show()
        return depth

dataset = Dataset(5000, 6 )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for i, data in enumerate(dataloader, 0):
    depth  = data




