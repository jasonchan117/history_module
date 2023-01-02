import sys

import tensorflow_datasets as tfds
import numpy as np
import cv2 as cv
import os
import quaternion

mode = 'test'

ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", shuffle_files=False)
ds_train = iter(tfds.as_numpy(ds['train']))
ds_test = iter(tfds.as_numpy(ds['test']))
dir = '/media/lang/My Passport/Dataset/MOvi/' + mode
cate = 14
cates = ["Action Figures", "Bag", "Board Games", "Bottles and Cans and Cups", "Camera", "Car Seat", "Consumer Goods", "Hat", "Headphones", "Keyboard", "Legos", "Media Cases", "Mouse", "None", "Shoe", "Stuffed Toys", "Toys"]



os.makedirs(os.path.join(dir, cates[cate]), exist_ok = True)
video_id = 0
cout = 0
while(True):

    cout += 1
    if cout >= len(ds[mode]):
        print('finish')
        break
    if mode == 'train':
        sample = next(ds_train)
    else:
        sample = next(ds_test)
    bboxes = sample["instances"]["bboxes"]
    bbox_frames = sample["instances"]["bbox_frames"]
    category = sample['instances']['category']
    centers = sample["instances"]["image_positions"]

    # if video_id <= 3030:
    #     if cate in category:
    #         video_id += 1
    #     print(video_id)
    #     continue
    if cate not in category:
        continue
    print('>>', video_id)
    os.makedirs(os.path.join(dir, cates[cate], str(video_id)), exist_ok=True)
    # Storaging the RGB

    b = []

    # for i in bbox_frames.numpy():
    #     a = []
    #     for j in i:
    #         a.append(j)
    #     b.append(a)

    # if video_id <= 3030:
    #     np.save(os.path.join(dir, cates[cate], str(video_id), 'mask.npy'), sample['segmentations'])
    # else:
    np.save(os.path.join(dir, cates[cate], str(video_id), 'mask.npy'), sample['segmentations'])
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'centers.npy'), centers)
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'bbox_frames_n.npy'), b)
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'category.npy'), category)
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'bboxes.npy'), bboxes)
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'depth_range.npy'), sample['metadata']['depth_range'])
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'bboxes_3d.npy'), sample['instances']['bboxes_3d'])
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'instances_r' + '.npy'), sample['instances']['quaternions'])
    # np.save(os.path.join(dir, cates[cate], str(video_id), 'instances_t' + '.npy'), sample['instances']['positions'])
    # for ind, i in enumerate(sample['video']):
    #     cv.imwrite(os.path.join(dir, cates[cate], str(video_id), 'rgb_' + str(ind) +'.png'), i)
    #     cv.imwrite(os.path.join(dir, cates[cate], str(video_id), 'depth_' + str(ind) + '.png'),  sample["depth"][ind])
    #     np.save(os.path.join(dir, cates[cate], str(video_id), 'cam_r_' + str(ind) +'.npy'),quaternion.as_rotation_matrix(
    #         np.quaternion(sample['camera']['quaternions'][ind][0],
    #                       sample['camera']['quaternions'][ind][1],
    #                       sample['camera']['quaternions'][ind][2],
    #                       sample['camera']['quaternions'][ind][3])))
    #     np.save(os.path.join(dir, cates[cate], str(video_id), 'cam_t_' + str(ind) +'.npy'), sample['camera']['positions'][ind])
    if video_id == 975:
        break
    video_id += 1
