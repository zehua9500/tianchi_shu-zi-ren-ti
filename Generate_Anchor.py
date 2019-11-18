import os
import pandas as pd
import numpy as np
import torch
from torchvision import models
import cv2
import random
import copy


def get_src_coordinate(pred, prior_anchor, op=torch):
    """
    :param pred: xywh
    :param prior_anchor: xywh
    :return: x1 y1 x2 y2
    """
    x = pred[:, 0] * prior_anchor[:, 2] + prior_anchor[:, 0]
    y = pred[:, 1] * prior_anchor[:, 3] + prior_anchor[:, 1]
    w = op.exp(pred[:, 2]) * prior_anchor[:, 2]
    h = op.exp(pred[:, 3]) * prior_anchor[:, 3]
    return op.stack((x, y, x + w, y + h), axis=1)


def generate_single_anchor(anchor_size):
    """
    根据anchor_size生成单个anchor
    anchor_size = [[15*1.3, 15*1.3],[15*1.3**2, 15*1.3**2],[15*1.3**3, 15*1.3**3]]
    """
    anchor_size = np.array(anchor_size) / 2  # 即anchor_size.shape = [anchor_nums,2]
    anchor = np.hstack((-1 * anchor_size, anchor_size))
    return anchor


def generate_prior_anchor(featshape, stride, anchor_size):
    """
    featshape = (h,w)
    生成特征图的先验框，prior_anchor shape = [h, w, anchor_nums, 4]，其值为bbox的左上角和右下角坐标，用于与GT做IOU匹配
    """
    single_anchor = generate_single_anchor(anchor_size)  # single_anchor shape = (anchor_nums, 4)
    anchor_nums = len(anchor_size)
    h, w = featshape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.reshape(-1, 1) * stride + stride / 2
    y = y.reshape(-1, 1) * stride + stride / 2  # 将x,y转为center point
    loc = np.hstack((x, y, x, y))[:, np.newaxis, :]  # loc shape转为（w*h, 1, 4）
    loc = loc + single_anchor
    loc = loc.reshape(-1, 4)

    loc[:, [0, 2]] = np.clip(loc[:, [0, 2]], 0, w * stride - 1)
    loc[:, [1, 3]] = np.clip(loc[:, [1, 3]], 0, h * stride - 1)
    return loc


def get_prior_anchor_separate(input_size):
    anchor_info = {
        "anchor_size_3": [[350, 350], [425, 425], [475, 475], [400, 300], [300, 400], [450, 350], [350, 450]],
        "anchor_size_1": [[20, 20], [30, 30], [45, 45], [70, 70], [100, 100], [100, 60], [60, 100]],
        "anchor_size_2": [[150, 150], [225, 225], [300, 300], [225, 150], [150, 225], [100, 200], [200, 100]],
        "feat_size_1": (input_size // 8, input_size // 8),
        "feat_size_2": (input_size // 16, input_size // 16),
        "feat_size_3": (input_size // 32, input_size // 32)
    }
    prior_anchor_1 = generate_prior_anchor(featshape=anchor_info["feat_size_1"], stride=8,
                                           anchor_size=anchor_info["anchor_size_1"])
    prior_anchor_2 = generate_prior_anchor(featshape=anchor_info["feat_size_2"], stride=16,
                                           anchor_size=anchor_info["anchor_size_2"])
    prior_anchor_3 = generate_prior_anchor(featshape=anchor_info["feat_size_3"], stride=32,
                                           anchor_size=anchor_info["anchor_size_3"])
    prior_anchor_1[:, 2:] -= prior_anchor_1[:, :2]
    prior_anchor_2[:, 2:] -= prior_anchor_2[:, :2]
    prior_anchor_3[:, 2:] -= prior_anchor_3[:, :2]
    return [prior_anchor_1, prior_anchor_2, prior_anchor_3]


def get_prior_anchor(input_size=800):
    """
    :param input_size:
    :return: x,y,w,h类型
    """
    prior_anchor_1, prior_anchor_2, prior_anchor_3 = get_prior_anchor_separate(input_size=input_size)
    prior_anchor = np.vstack((prior_anchor_1, prior_anchor_2, prior_anchor_3))
    prior_anchor = torch.from_numpy(prior_anchor).cuda().float()
    prior_anchor[:, 2:] -= prior_anchor[:, :2]
    return prior_anchor


prior_anchor = get_prior_anchor()


class Generate_Anchor():
    def __init__(self, batch_size, stride, feat_size, anchor_size, thres_hold=0.5):
        """
        feat_size = (h,w)
        """
        self.batch_size = batch_size
        self.stride = stride
        self.prior_anchor = generate_prior_anchor(feat_size, stride, anchor_size)
        self.thres_hold = thres_hold
        # prior_anchor存放的是特征图相应原图的坐标，其shape=[w*h*anchor_nums, 4]

    def coordinate_change(self, anchor_loc, bbox_loc, op=np):
        """
        shape均为 [-1,4] [x1,y1,x2,y2] op存放操作，np或torch
        """
        anchor_loc[:, 2:] -= anchor_loc[:, :2]
        bbox_loc[:, 2:] -= bbox_loc[:, :2]  # 坐标转换为x,y,w,h  左上角坐标
        tx = (bbox_loc[:, 0] - anchor_loc[:, 0]) / anchor_loc[:, 2]
        ty = (bbox_loc[:, 1] - anchor_loc[:, 1]) / anchor_loc[:, 3]
        tw = op.log(bbox_loc[:, 2] / anchor_loc[:, 2])
        th = op.log(bbox_loc[:, 3] / anchor_loc[:, 3])
        return op.stack((tx, ty, tw, th), axis=1)

    def getAnchor(self, bboxs, use_cuda=True):
        """
        bboxs = [batch_size, anchor_nums, 4]  当多分类时bboxs = [batch_size, anchor_nums, 5],类别标签从1开始 非0
        """
        Label_valid, Loc, Cls, Loc_valid = [], [], [], []
        for i in range(self.batch_size):  # 注意 当bbox为空时可能出问题
            bbox = bboxs[i]
            bbox = bbox[bbox[:, 0] >= 0]  # anchor_nums,4
            cls_save = bbox[:, -1]  # 当bbox输入有类别信息时，cls存储类别信息
            bbox = bbox[:, :4]
            prior_anchor = copy.copy(self.prior_anchor)[:, np.newaxis, :]
            x_tl = np.maximum(prior_anchor[:, :, 0], bbox[:, 0])
            y_tl = np.maximum(prior_anchor[:, :, 1], bbox[:, 1])
            x_br = np.minimum(prior_anchor[:, :, 2], bbox[:, 2])
            y_br = np.minimum(prior_anchor[:, :, 3], bbox[:, 3])
            i = np.clip(x_br - x_tl, 0, None) * np.clip(y_br - y_tl, 0, None)
            u = ((prior_anchor[:, :, 2] - prior_anchor[:, :, 0]) * (prior_anchor[:, :, 3] - prior_anchor[:, :, 1])) + \
                (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            iou = i / (u - i + 1e-5)
            cls_index = iou.argmax(axis=1)

            # 有些GT 被分成了2个 bbox，因此让anchor与多个gt的交集大于self.thres_hold即为正例
            # 回归目标为 iou大的那一个
            max_iou_value = iou[np.arange(iou.shape[0]), cls_index]  # 最大iou值对应cls
            single_iou_index = (max_iou_value > self.thres_hold)
            mutil_iou_index = (np.sum(iou,axis=1) > self.thres_hold) * (~single_iou_index)
            iou_index = single_iou_index + mutil_iou_index
            #print("mutil_iou_index, ", np.sum(mutil_iou_index) / mutil_iou_index.shape[0])

            label_valid = (max_iou_value < 0.1) + iou_index
            loc = self.coordinate_change(np.squeeze(prior_anchor), bbox[cls_index])  # * label[:, np.newaxis]
            # loc = bbox[cls_index] * label[:,np.newaxis]
            Label_valid.append(label_valid)
            Loc_valid.append(single_iou_index)
            Loc.append(loc)
            Cls.append(cls_save[cls_index] * iou_index)  # size = w*h*anchor_nums
        # return torch.cat(Cls).cuda(), torch.cat(Label_valid).cuda(), torch.cat(Loc).cuda().float()
        return torch.from_numpy(np.hstack(Cls)).cuda().float(), \
               torch.from_numpy(np.hstack(Label_valid)).cuda(), \
               torch.from_numpy(np.vstack(Loc)).cuda().float(), \
               torch.from_numpy(np.hstack(Loc_valid)).cuda()
