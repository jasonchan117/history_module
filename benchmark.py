import os
import glob
import numpy as np
import math
import _pickle as cPickle




def compute_3d_iou_new(RT_1, RT_2, noc_cube_1, noc_cube_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''

    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2):
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    symmetry_flag = False
    # if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (
    #         class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility == 0):
    #
    #     bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)
    #
    #     def y_rotation_matrix(theta):
    #         return np.array([[np.cos(theta), 0, np.sin(theta), 0],
    #                          [0, 1, 0, 0],
    #                          [-np.sin(theta), 0, np.cos(theta), 0],
    #                          [0, 0, 0, 1]])
    #
    #     n = 20
    #     max_iou = 0
    #     for i in range(n):
    #         rotated_RT_1 = RT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
    #         max_iou = max(max_iou,
    #                       asymmetric_3d_iou(rotated_RT_1, RT_2, noc_cube_1, noc_cube_2))
    # else:
    max_iou = asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2)

    return max_iou


def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

# bottle, bowl, camera, can, laptop, mug
def compute_RT_degree_cm_symmetry(RT_1, RT_2):
    if RT_1 is None or RT_2 is None:
        return 10000, 10000
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        return 10000, 10000

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # if synset_names[class_id] in ['bottle', 'can', 'bowl']:
    #     y = np.array([0, 1, 0])
    #     y1 = R1 @ y
    #     y2 = R2 @ y
    #     theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # elif synset_names[class_id] == 'mug' and handle_visibility == 0:
    #     y = np.array([0, 1, 0])
    #     y1 = R1 @ y
    #     y2 = R2 @ y
    #     theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
    #     y_180_RT = np.diag([-1.0, 1.0, -1.0])
    #     R = R1 @ R2.transpose()
    #     R_rot = R1 @ y_180_RT @ R2.transpose()
    #     theta = min(np.arccos((np.trace(R) - 1) / 2),
    #                 np.arccos((np.trace(R_rot) - 1) / 2))
    # else:
    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2)
    result = np.array([theta, shift])

    return result


score_dict = {}


def benchmark(pred_pose, gt_pose, bbox):

    score = 0
    score_25 = 0
    rot_err = 0
    trans_err = 0

    cls_in_5_5 = 0
    cls_iou_25 = 0

    cls_rot = []
    cls_trans = []


    # z_180_RT = np.zeros((4, 4), dtype=np.float32)
    # z_180_RT[:3, :3] = np.diag([-1, -1, 1])
    # z_180_RT[3, 3] = 1
    # pred_pose = z_180_RT @ pred_pose
    gt_pose = np.array(gt_pose)
    pred_pose = np.array(pred_pose)

    result = compute_RT_degree_cm_symmetry(pred_pose, gt_pose)
    miou = compute_3d_iou_new(gt_pose, pred_pose, bbox, bbox)

    if miou > 0.25 and result[0] < 360:
        cls_rot.append(result[0])
    if miou > 0.25:
        cls_trans.append(result[1])
    if miou > 0.25:
        cls_iou_25 = cls_iou_25 + 1
    if result[0] < 5 and result[1] < 50:
        cls_in_5_5 = cls_in_5_5 + 1

    score = score + cls_in_5_5
    score_25 = score_25 + cls_iou_25
    if cls_rot == []:
        rot_err = None
    else:
        rot_err = rot_err + np.mean(cls_rot)

    if cls_trans == []:
        trans_err = None
    else:
        trans_err = trans_err + np.mean(cls_trans)


    return score, score_25, rot_err, trans_err



