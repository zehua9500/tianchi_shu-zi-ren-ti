import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import os
import time
import random
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from Generate_Anchor import get_prior_anchor_separate, get_src_coordinate, Generate_Anchor
from Data_PreProcess import Data_preprocess as dp
import copy


class ResNet(torch.nn.Module):
    def __init__(self, model=torchvision.models.resnet50()):
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = model.layer1
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False)
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.squeeze2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.squeeze3 = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.squeeze4 = nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [self.squeeze2(layer2), self.squeeze3(layer3), self.squeeze4(layer4)]


class FPN(torch.nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # self.upsample1 = F.upsample(input, size=None, scale_factor=None,mode='nearest', align_corners=None)
        self.head = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, layer2, layer3, layer4):
        layer4_upsample = F.interpolate(layer4, scale_factor=2)
        layer3 = layer3 + layer4_upsample

        layer3_upsample = F.interpolate(layer3, scale_factor=2)
        layer2 = layer2 + layer3_upsample
        return [self.head(layer2), self.head(layer3), self.head(layer4)]


class S2(torch.nn.Module):
    def __init__(self, cls_num, feat_shape=(256, 7, 7)):
        super(S2, self).__init__()
        self.cls_num = cls_num + 1
        self.feat_shape = feat_shape
        self.cls_feature = torch.nn.Sequential(
            torch.nn.Linear(256 * 7 * 7, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True))

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(2048, self.cls_num),
            torch.nn.Sigmoid())

        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(256 * 7 * 7, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 4))

    def getbalanceweight(self, cls_gt, smooth=0.5):
        # balanceweight = torch.ones_like(cls_gt)
        cls_num = torch.ones(self.cls_num, device="cuda:0") * smooth
        total_num = cls_gt.size(0)
        for i in range(self.cls_num):
            cls_num[i] += torch.sum(cls_gt == i)
        cls_weight = total_num / cls_num
        cls_weight = cls_weight / torch.mean(cls_weight)
        # cls_weight = torch.clamp(cls_weight,0.1,10)
        return cls_weight[cls_gt]

    def forward(self, x, cls_gt=None, loc_gt=None, rpn_loc=None, valid_loc=None, threshold=0.4, eps=1e-7):
        """"
        test时 batch=1
        """
        x = x.contiguous().view(x.size(0), -1)

        feature = self.cls_feature(x)
        cls = self.cls(feature)

        loc = self.loc_layer(x)
        score, indices = torch.max(cls, dim=1)
        if not self.training:
            index = (score > threshold) * (indices == 1)
            #index = (indices == 1)
            cls = cls[torch.arange(cls.size(0)), indices][index]
            loc = loc[index]
            rpn_loc = rpn_loc[index]
            rpn_loc[:, 2:4] -= rpn_loc[:, :2]
            # return cls, get_src_coordinate(loc, rpn_loc)
            return torch.cat((get_src_coordinate(loc, rpn_loc), cls.view(-1, 1)), dim=1)
        loc_loss = dp.smooth_L1(loc, loc_gt.cuda(), valid_loc)  # 滤去难负例
        cls_gt = cls_gt.long().cuda()  # [valid_index]
        #cls = cls  # [valid_index]
        cls_gt_onehot = dp.onehot(cls_gt, self.cls_num, smooth=0).cuda().float()
        # print("loc_gt \n ", loc_gt)
        # print("cls_gt \n ", cls_gt)
        # print("cls_gt_onehot \n ", cls_gt_onehot)
        balance_weight = self.getbalanceweight(cls_gt).view(-1, 1)
        CE_loss = -1 * (cls_gt_onehot * torch.log(cls + eps) + (1 - cls_gt_onehot) * torch.log(1 + eps - cls)) * balance_weight * ((cls_gt_onehot - cls) ** 2)

        pos_feature = feature[cls_gt == 1].view(-1, 2048)
        hard_neg_fature = feature[cls_gt == 2].view(-1, 2048)
        # neg_feature = feature[cls_gt==0].view(-1,2048)
        # print("feature pos ", pos_feature.shape)
        pos_feature = torch.nn.functional.normalize(pos_feature, dim=1, p=2)
        hard_neg_fature = torch.nn.functional.normalize(hard_neg_fature, dim=1, p=2)
        # neg_feature = torch.nn.functional.normalize(neg_feature, dim=1, p=2)

        if (torch.sum(cls_gt == 1) > 0 and torch.sum(cls_gt == 2) > 0):
            pos_hard_matrix = torch.sum(pos_feature.unsqueeze(1) * hard_neg_fature.unsqueeze(0), dim=2)  # cos(θ)
        else:
            pos_hard_matrix = torch.tensor(-10, dtype=torch.float, device="cuda:0")
        if torch.sum(cls_gt == 1) > 0:
            pos_pos_matrix = -1 * torch.sum(pos_feature.unsqueeze(1) * pos_feature.unsqueeze(0), dim=2)
        else:
            pos_pos_matrix = torch.tensor(-10, dtype=torch.float, device="cuda:0")
        if torch.sum(cls_gt == 2) > 0:
            hard_hard_matrix = -1 * torch.sum(hard_neg_fature.unsqueeze(1) * hard_neg_fature.unsqueeze(0), dim=2)
        else:
            hard_hard_matrix = torch.tensor(-10, dtype=torch.float, device="cuda:0")
        return torch.mean(CE_loss), loc_loss, torch.mean(torch.exp(pos_hard_matrix)), torch.mean(torch.exp(pos_pos_matrix)), torch.mean(
            torch.exp(hard_hard_matrix))


class RPN(torch.nn.Module):
    a = torch.ones(1).cuda().float()

    def __init__(self, anchor_num, input_size, is_training=True):
        super(RPN, self).__init__()
        # self.a = torch.ones(1).cuda().float()
        self.count = 0
        self.input_size = input_size
        self.cls_layer = torch.nn.Sequential(
            nn.Conv2d(256, anchor_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Sigmoid())
        self.loc_layer = torch.nn.Sequential(
            nn.Conv2d(256, 4 * anchor_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))

    def get_roi(self, featureMap, coordinate, roi_size=(7, 7), rois_num=10):
        """
        :param featureMap: 无batch_size，   shape = c,h,w
        :param coordinate:
        :return:
        """
        # valid_index = [True] * coordinate.size(0)
        # coordinate[:, :2] = torch.round(coordinate[:, :2])
        # coordinate[:, 2:4] = torch.ceil(coordinate[:, 2:4])
        # print("coordinate", coordinate)
        # coordinate = coordinate.int()
        # coordinate = coordinate.clamp(0, featureMap.shape[1] - 1)
        # coordinate[:,:2] -= 1
        # coordinate[:, 2:4] += 1
        # valid_index = (coordinate[:, 2] - coordinate[:, 0] >= 1) * (coordinate[:, 3] - coordinate[:, 1] >= 1)
        # print("valid_index ",torch.sum(valid_index))
        if (coordinate.size(0) == 0):
            return torch.Tensor(0, 256, 7, 7).cuda().float()
        # coordinate = coordinate[valid_index]
        rois = []
        # nums = rois_num if rois_num < coordinate.shape[0] else coordinate.shape[0]
        for i in range(coordinate.shape[0]):
            loc = coordinate[i]
            # print("loc ",loc)
            roi = featureMap[:, loc[1]:loc[3], loc[0]:loc[2]]
            # print("roi = ",roi)
            rois.append(F.adaptive_max_pool2d(roi, roi_size))
        return torch.stack(rois, dim=0)

    def coordinate_change(self, loc_rpn, loc_gt, op=torch):
        """
        :param loc_rpn:  rpn回归和prior_anchor得到的loc预测绝对值  x1 y1 x2 y2 score
        :param loc_gt: x y x y
        :return: 2阶段的回归目标值
        """
        loc_rpn[:, 2:4] -= loc_rpn[:, :2]
        loc_gt[:, 2:4] -= loc_gt[:, :2]
        tx = (loc_gt[:, 0] - loc_rpn[:, 0]) / loc_rpn[:, 2]
        ty = (loc_gt[:, 1] - loc_rpn[:, 1]) / loc_rpn[:, 3]
        tw = op.log(loc_gt[:, 2] / loc_rpn[:, 2])
        th = op.log(loc_gt[:, 3] / loc_rpn[:, 3])
        return op.stack((tx, ty, tw, th), axis=1)

    def get_s2_target(self, rpn_loc, bboxs_gt):
        """
        与gt的iou>0.4的为正例，类别为原类别 此处就1类即为1
        与gt的iou<0.4但是 score>0.8的为难负例 类别类类别数+1 即为2
        其余的为负例 类别为0
        注：loc回归需滤去难负例
        :param rpn_loc: numpy类型 shape=（N,5),第五列为 rpn预测的score 已经过score > thres和nms处理
        :param bboxs_gt: numpy类型 shape=（N,5),第五列为 类别
        :return:
        """
        bboxs_gt = bboxs_gt[bboxs_gt[:, 4] >= 0]
        # cls = torch.from_numpy(bboxs_gt[:,-1]).cuda()
        iou = dp.get_iouMatrix(rpn_loc, bboxs_gt)
        iou = torch.from_numpy(iou).cuda()
        bboxs_gt = torch.from_numpy((bboxs_gt)).cuda().float()
        cls = bboxs_gt[:, -1]
        iou_value, iou_index = iou.max(dim=1)
        # s2_cls = cls[iou_index] * (iou_value > 0.55).float()
        # pos_index = (torch.sum(iou, dim=1) > 0.35).float()
        single_iou_index = (iou_value > 0.45)
        mutil_iou_index = (torch.sum(iou, axis=1) > 0.45) * (~single_iou_index)
        pos_index = single_iou_index + mutil_iou_index
        neg_index = (torch.sum(iou, dim=1) < 0.10).float()
        # valid_index = pos_index + neg_index
        rpn_loc = torch.from_numpy(rpn_loc).cuda().float()
        # rpn_loc = torch.tensor(rpn_loc, device="cuda:0", dtype=torch.float, requires_grad=False)
        score = rpn_loc[:, -1]
        hard_neg_index = neg_index * (score > 0.8).float()
        # pos_num = torch.min(torch.sum(pos_index), torch.tensor(16, device="cuda:0", dtype=torch.float)).long()
        # hard_neg_num = torch.min(torch.sum(neg_index), torch.tensor(16, device="cuda:0", dtype=torch.float)).long()
        # neg_num = 48 - pos_num - hard_neg_num
        # index1 = (torch.sort(score * pos_index, descending=True).indices < pos_num)
        # index2 = (torch.sort(score * hard_neg_index, descending=True).indices < hard_neg_num)
        # index3 =  (torch.sort(score * (neg_index - hard_neg_index), descending=True).indices < neg_num)
        # nums_balance_index = index1 + index2 + index3
        # nums_balance_index = (torch.sort(score * pos_index, descending=True).indices < 16) + \
        #                     (torch.sort(score * hard_neg_index, descending=True).indices < 16) + \
        #                    (torch.sort(score * (neg_index - hard_neg_index), descending=True).indices < 24)
        s2_cls = cls[iou_index] * pos_index.float() + hard_neg_index * 2
        s2_loc = self.coordinate_change(rpn_loc, bboxs_gt[iou_index])
        return s2_cls, s2_loc, single_iou_index  # , nums_balance_index

    def forward(self, featureMap, prior_anchor=None, stride=None, cls_gt=None, gt_bboxs=None, threshold=0.5, roi_size=(7, 7)):
        """
        :param featureMap:
        :param prior_anchor: 为单batch的 prior_anchor
        :param cls:
        :param gt_bboxs: 为x1 y1 x2 y2类型
        :return:
        """
        cls = self.cls_layer(featureMap).permute(0, 2, 3, 1).contiguous().view(-1)
        loc = self.loc_layer(featureMap).permute(0, 2, 3, 1).contiguous().view(-1, 4)
        # if not self.training:
        #    return cls, loc
        batch_size = featureMap.size(0)
        anchor_num = prior_anchor.size(0)
        rois = []
        stage2_cls = []
        stage2_valid_index = []
        stage2_loc = []
        stage2_score = []
        stage2_numIndex = []
        eval_test = []
        for b in range(batch_size):
            cls_score = cls[b * anchor_num:(b + 1) * anchor_num]
            index = cls_score > threshold

            src_coordinate = get_src_coordinate(loc[b * anchor_num:(b + 1) * anchor_num][index], prior_anchor[index])  # 坐标还原x y x y
            src_coordinate = torch.cat((src_coordinate, cls_score.view(-1, 1)[index]), dim=1)  # 坐标为在原图上的坐标
            src_coordinate = src_coordinate.clamp(0, self.input_size)  # input为方形的。此处先对宽高做相同处理
            valid_index = (src_coordinate[:, 2] - src_coordinate[:, 0] >= 2 * stride) * (
                    src_coordinate[:, 3] - src_coordinate[:, 1] >= 2 * stride)
            src_coordinate = src_coordinate[valid_index]
            # src_coordinate = src_coordinate.detach().cpu().numpy()  # 此后src_coordinate为numpy
            index_nms = dp.py_cpu_nms(src_coordinate.detach().cpu().numpy(), thresh=0.3)[:32]  # 后续改GPU的nms
            src_coordinate = src_coordinate[index_nms]
            if (src_coordinate.size(0) == 0): continue
            # roi_start = time.time()
            rois.append(self.get_roi(featureMap[b], src_coordinate.int() / stride, roi_size=roi_size))
            # print("roi : %.5f",time.time() - roi_start)
            if not self.training:
                eval_test.append(src_coordinate)
                continue
                # print("src ", src_coordinate)
            # else:
            # s2_cls, s2_loc = self.get_s2_target(src_coordinate.detach().cpu().numpy(), gt_bboxs[b])  # 对应rois
            """
            pos_nums = 16 if torch.sum(s2_cls == 1) > 16 else torch.sum(s2_cls == 1)
            hard_neg = 16 if torch.sum(s2_cls == 2) > 16 else torch.sum(s2_cls == 2)
            neg_num = (64 if s2_cls.size(0) > 64 else s2_cls.size(0)) - pos_nums - hard_neg
            index = (torch.sort(cls_score * (s2_cls == 1).float(), descending=True).indices < pos_nums) + \
                    (torch.sort(cls_score * (s2_cls == 2).float(), descending=True).indices < hard_neg) + \
                    (torch.sort(cls_score * (s2_cls == 0).float(), descending=True).indices < neg_num)
            src_coordinate = src_coordinate[index]
    
            """
            s2_cls, s2_loc, s2_loc_valid = self.get_s2_target(src_coordinate.detach().cpu().numpy(), gt_bboxs[b])  # 对应rois
            cls_score = cls_score[index][valid_index][index_nms]  # 依次滤掉score低的，无效roi和nms,
            stage2_score.append(cls_score)
            stage2_cls.append(s2_cls)
            stage2_loc.append(s2_loc)
            stage2_valid_index.append(s2_loc_valid)
        if not self.training:
            rois = torch.Tensor(0, 256, 7, 7).cuda().float() if len(rois) == 0 else torch.cat(rois, dim=0)
            eval_test = torch.Tensor(0, 5).cuda().float() if len(eval_test) == 0 else torch.cat(eval_test, dim=0)
            return eval_test, rois

        if (len(rois) == 0):
            return cls, loc, torch.Tensor(0, 256, 7, 7).cuda().float(), [], [], [], []
        return cls, loc, torch.cat(rois, dim=0), \
               torch.cat(stage2_cls, dim=0), \
               torch.cat(stage2_loc, dim=0), \
               torch.cat(stage2_score, dim=0), \
               torch.cat(stage2_valid_index, dim=0)
        # 返回分类和回归的预测值，用于计算loss 和rois,rois的分类和回归的target值


class Net(torch.nn.Module):
    def __init__(self, anchor_num=7, input_size=800, is_training=True, stride=[8, 16, 32], anchor_info=None,
                 batch_size=None, anchor_thres=0.35):
        super(Net, self).__init__()
        self.prior_anchor = get_prior_anchor_separate(input_size=input_size) # 由于切割的图都是相同大小，对于每个batch的prior anchor都是一样的
        self.prior_anchor = [torch.from_numpy(anchor).cuda().float().detach() for anchor in
                             self.prior_anchor]  # prior_anchor是单batch的，不包含batch_size
        self.resnet = ResNet()
        self.fpn = FPN()
        self.is_training = is_training
        self.rpn = RPN(anchor_num=anchor_num, input_size=input_size, is_training=is_training)
        self.stride = stride
        self.stage_2 = S2(2, feat_shape=(256, 7, 7))
        if is_training:
            self.anchor_generater = [
                Generate_Anchor(batch_size=batch_size, stride=stride[0], feat_size=(input_size // 8, input_size // 8),
                                anchor_size=anchor_info["anchor_size_1"], thres_hold=anchor_thres),
                Generate_Anchor(batch_size=batch_size, stride=stride[1], feat_size=(input_size // 16, input_size // 16),
                                anchor_size=anchor_info["anchor_size_2"], thres_hold=anchor_thres),
                Generate_Anchor(batch_size=batch_size, stride=stride[2], feat_size=(input_size // 32, input_size // 32),
                                anchor_size=anchor_info["anchor_size_3"], thres_hold=anchor_thres)]

    def forward(self, x, gt_bboxs=None, roi_maxnums=160, threshold=0.5, s2_threshold=0.4):
        """
        写的混乱，过后再整理吧
        :param x:  cuda类型 img数据
        :param gt_bboxs:  [batch_size, nums, 5]  5 = x1,y1,x2,y2,cls
        :return:
        """
        start = time.time()
        layer2, layer3, layer4 = self.resnet(x)
        if not self.training:
            cls, loc, evals, s2_res = [], [], [], []
            for idx, layer in enumerate(self.fpn(layer2, layer3, layer4)):
                _eval, _rois = self.rpn(layer, self.prior_anchor[idx], self.stride[idx], threshold=threshold)
                evals.append(_eval)
                if (_rois.size(0) != 0):
                    _s2_res = self.stage_2(_rois, rpn_loc=_eval[:, :4], threshold=s2_threshold)
                    s2_res.append(_s2_res)
            return torch.Tensor(0, 5).cuda().float() if len(s2_res) == 0 else torch.cat(s2_res, dim=0)
                    #torch.Tensor(0, 5).cuda().float() if len(evals) == 0 else torch.cat(evals, dim=0)
                # index = _cls > threshold
                # cls.append(_cls[index])
                # _loc = get_src_coordinate(_loc, self.prior_anchor[idx])
                # loc.append(get_src_coordinate(_loc[index], self.prior_anchor[idx][index]))
                # loc.append(_loc)
            # return torch.cat(cls), torch.cat(loc, dim=0)
            #return torch.Tensor(0, 5).cuda().float() if len(evals) == 0 else torch.cat(evals, dim=0), \
            #return  torch.Tensor(0, 5).cuda().float() if len(s2_res) == 0 else torch.cat(s2_res, dim=0)
        # training
        rpn_cls_loss, rpn_loc_loss, s2_cls_loss, s2_loc_loss = torch.tensor(0.0).cuda(), \
                                                               torch.tensor(0.0).cuda(), \
                                                               torch.tensor(0.0).cuda(), \
                                                               torch.tensor(0.0).cuda()
        cls, loc, rois, rpn_cls_gt, rpn_loc_gt, rpn_valid_gt, s2_score, s2_cls_gt, s2_loc_gt, s2_valid_index = [], [], [], [], [], [], [], [], [], []
        rois_cls, rois_loc, rois_score = [], [], []
        rpn_loc_valid, s2_valid_loc = [], []
        for idx, layer in enumerate(self.fpn(layer2, layer3, layer4)):  # fpn输出
            _cls_gt, _valid_gt, _loc_gt, _loc_valid = self.anchor_generater[idx].getAnchor(gt_bboxs)  # ground_truth生成 全放Net里而不放main里。 方便fpn各层独立
            _cls, _loc, _rois, _rois_cls, _rois_loc, _rois_score, _rois_loc_valid = self.rpn(layer, self.prior_anchor[idx], self.stride[idx], cls_gt=_cls_gt,
                                                                                             gt_bboxs=gt_bboxs)
            # _cls, _loc = self.roi(layer, self.prior_anchor[idx], self.stride[idx])
            cls.append(_cls)
            loc.append(_loc)
            # rpn_cls_loss += dp.focal_loss(_cls, _cls_gt, is_valid=_valid_gt) * (2 ** idx)
            # rpn_loc_loss += 0 if torch.sum(_cls_gt > 0) == 0 else dp.smooth_L1(_loc, _loc_gt, _cls_gt > 0) * (2 ** idx)
            rpn_cls_gt.append(_cls_gt)
            rpn_loc_gt.append(_loc_gt)
            rpn_valid_gt.append(_valid_gt)
            rpn_loc_valid.append(_loc_valid)
            if _rois.size(0) == 0: continue
            rois.append(_rois)
            rois_cls.append(_rois_cls)
            rois_loc.append(_rois_loc)
            rois_score.append(_rois_score)
            s2_valid_loc.append(_rois_loc_valid)
            # s2_valid_index.append(_rois_valid)
            """
            if (_rois.size(0) == 0): continue
            _, indices = torch.sort(_rois_score)
            indices = indices.flip(dims=(0,))[:roi_maxnums]

            _s2_cls_loss, _s2_loc_loss = self.stage2(_rois[indices], cls_gt=_rois_cls[indices], loc_gt=_rois_loc[indices])
            s2_cls_loss += _s2_cls_loss * (2 ** idx)
            s2_loc_loss += _s2_loc_loss * (2 ** idx)  #高层经过maxpooling后，梯度回传的影响会不会小于低层fpn ?
            """
            # rois.append(_rois)
        rpn_cls_loss = dp.focal_loss(torch.cat(cls), torch.cat(rpn_cls_gt), is_valid=torch.cat(rpn_valid_gt))
        rpn_loc_loss = dp.smooth_L1(torch.cat(loc, dim=0), torch.cat(rpn_loc_gt), torch.cat(rpn_loc_valid))
        if len(rois) > 0:
            _, indices = torch.sort(torch.cat(rois_score))
            indices = indices.flip(dims=(0,))[:roi_maxnums]
            s2_cls_loss, s2_loc_loss, pos_hard, pos_pos, hard_hard = self.stage_2(torch.cat(rois)[indices], cls_gt=torch.cat(rois_cls)[indices],
                                                                                  loc_gt=torch.cat(rois_loc)[indices],
                                                                                  valid_loc=torch.cat(s2_valid_loc)[indices],
                                                                                  threshold=s2_threshold)
        return rpn_cls_loss, rpn_loc_loss, s2_cls_loss, s2_loc_loss, pos_hard, pos_pos, hard_hard
