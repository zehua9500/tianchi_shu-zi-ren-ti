import cv2
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import time
import copy
import torch


class Data_preprocess():
    @staticmethod
    def draw_bbox(img, bbox, labelName=None, color=(0, 0, 255)):  # BGR
        # bbox坐标为左上右下
        img = copy.copy(img)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if not labelName is None:
            img = cv2.putText(img, str(labelName), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 150, 190),
                              1)
        return img

    @staticmethod
    def draw_bboxs(img, bboxs, color=(0, 0, 255)):  # BGR
        # bbox坐标为左上右下
        # img = copy.copy(img)
        flag = bboxs.shape[1] >= 5
        if flag:
            scores = bboxs[:, 4]
        bboxs = bboxs[:, :4].astype(np.int)
        for i in range(bboxs.shape[0]):
            bbox = bboxs[i]
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            if flag:
                img = cv2.putText(img, str(scores[i]), (bbox[0] + 2, bbox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  color, 1)
        return img

    @staticmethod
    def imgDisplay(img, imgname="img", destroy=True, auto=0):
        cv2.namedWindow(imgname, auto)
        cv2.imshow(imgname, img)
        cv2.waitKey(0)
        if (destroy):
            cv2.destroyAllWindows()

    @staticmethod
    def onehot(label, clsNum, smooth=0, op=torch):
        # label起始为0
        nums = len(label)
        smooth_weight = smooth / clsNum
        onehotLabel = op.ones((nums, clsNum)) * smooth_weight
        onehotLabel[op.arange(nums), label] += 1 - smooth
        return onehotLabel
        """
        if smooth:
            smooth_weight = 0.1 / clsNum
            onehotLabel = np.ones((nums, clsNum)) * smooth_weight
            onehotLabel[np.arange(nums), label] += 0.9
        else:
            onehotLabel = np.zeros((nums, clsNum))
            onehotLabel[np.arange(nums), label] = 1
        return onehotLabel
        """

    @staticmethod
    def getBalanceWeight(typeList, clsNum, eps=1e-5):
        typeList = np.array(typeList)
        balanceNum = np.zeros(clsNum)
        for t in range(clsNum):
            balanceNum[t] = np.sum(typeList == t)
        # mean = np.mean(balanceNum)
        return np.mean(balanceNum) / (balanceNum + eps)

    @staticmethod
    def getImgSize(imgfoldPath):
        """
        :param imgfoldPath:  存放图片的文件夹
        :return:  返回pd_DataFrams文件，其列为FIleName，width，height，ratio
        """
        imglist_src = pd.Series(os.listdir(imgfoldPath))
        imglist = pd.Series([imgfoldPath + '\\'] * len(imglist_src)) + imglist_src
        width = []
        height = []
        for imgName in imglist:
            img = cv2.imread(imgName)
            h, w = img.shape[:2]
            width.append(w)
            height.append(h)
        Info = pd.DataFrame({"FileName": imglist_src, "width": width, "height": height})
        Info["ratio"] = Info["width"] / Info["height"]
        return Info

    @staticmethod
    def knnSeekAnchor(hw, anchor_num=9, iter_num=50):
        """
        :param hw: numpy文件，shape= [N,2] 存放hw,N为bbox数
        :param anchor_num: 聚类中心数
        :param iter_num:
        :return: 聚类中心的h和w列表
        """
        max_h, min_h = np.max(hw[:, 0]), np.min(hw[:, 0])
        base_h = [h for h in range(min_h, max_h, (max_h - min_h) // (anchor_num - 1))]
        base_h = np.array(base_h)

        max_w, min_w = np.max(hw[:, 1]), np.min(hw[:, 1])
        base_w = [w for w in range(min_w, max_w, (max_w - min_w) // (anchor_num - 1))]
        base_w = np.array(base_w)
        # base_hw = np.array([base_h,base_w]).T

        h = np.array(hw[:, 0])[:, np.newaxis]
        w = np.array(hw[:, 1])[:, np.newaxis]
        for i in range(iter_num):
            i_h = np.minimum(h, base_h)
            i_w = np.minimum(w, base_w)
            i_area = i_h * i_w
            u_area = h * w + base_h * base_w
            iou = i_area / (u_area - i_area + 1e-5)
            index = np.argmax(iou, axis=1)
            for k in range(anchor_num):
                if (np.sum(index == k) == 0): continue
                base_h[k] = np.mean(h[index == k])
                base_w[k] = np.mean(w[index == k])
        return base_h, base_w

    @staticmethod
    def draw_distribution(pos_w, pos_h, anchor_w, anchor_h):
        """
        :param pos_w:
        :param pos_h: pos样本宽高
        :param anchor_w:
        :param anchor_h: anchor宽高（聚类得到）
        :return: 画出其分布
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Result Analysis')
        ax1.set_xlabel('width')
        ax1.set_ylabel('height')
        ax1.scatter(pos_w, pos_h, s=25, marker='.')
        ax1.scatter(anchor_w, anchor_h, s=25, c='r', marker='.')
        plt.show()

    @staticmethod
    def csv2dict(label, filename="filename"):
        gt = {}
        for idx in range(len(label)):
            info = label.iloc[idx]
            if info[filename] not in gt:
                gt[info[filename]] = []
            gt[info[filename]].append([info["x1"], info["y1"], info["x2"], info["y2"]])
        return gt

    @staticmethod
    def nms(bbox, threshold=0.3, eps=1e-5):
        # bbox[i] = [x1, y1, x2, y2, score]
        """
        一次性计算所有 iou 再循环。  当bbox过多时占内存大
        :param bbox:
        :param threshold:
        :param eps:
        :return:
        """
        index = np.argsort(bbox[:, -1])[::-1]
        bbox = bbox[index]
        res = []  # 保存index
        i_x1 = np.maximum(bbox[:, 0].reshape(-1, 1), bbox[:, 0])
        i_y1 = np.maximum(bbox[:, 1].reshape(-1, 1), bbox[:, 1])

        i_x2 = np.minimum(bbox[:, 2].reshape(-1, 1), bbox[:, 2])
        i_y2 = np.minimum(bbox[:, 3].reshape(-1, 1), bbox[:, 3])

        i = np.maximum(0, i_x2 - i_x1 + 1) * np.maximum(0, i_y2 - i_y1 + 1)
        area = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
        u = area + area.reshape(-1, 1)
        iou = i / (u - i + eps)  # 此时iou.shape = [length, length]
        while index.shape[0]:
            res.append(index[0])
            keepIndex = iou[0, :] < threshold  # iou小于阈值的保留
            index = index[keepIndex]
            iou = iou[keepIndex][:, keepIndex]
        return res

    @staticmethod
    def py_cpu_nms(dets, thresh=0.3):
        """
        参考 CSDN
        :param dets:
        :param thresh:
        :return:
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]
            keep.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    @staticmethod
    def get_iouMatrix(vec1, vec2):
        """
        其shape均为【-1，N】 N>=4, 前4列为 x1,y1,x2,y2
        """
        vec1 = vec1[:, :4][:, np.newaxis]
        vec2 = vec2[:, :4]
        x_tl = np.maximum(vec1[:, :, 0], vec2[:, 0])
        y_tl = np.maximum(vec1[:, :, 1], vec2[:, 1])
        x_br = np.minimum(vec1[:, :, 2], vec2[:, 2])
        y_br = np.minimum(vec1[:, :, 3], vec2[:, 3])

        i = np.clip(x_br - x_tl + 1, 0, None) * np.clip(y_br - y_tl + 1, 0, None)
        u = (vec1[:, :, 2] - vec1[:, :, 0] + 1) * (vec1[:, :, 3] - vec1[:, :, 1] + 1) + \
            (vec2[:, 2] - vec2[:, 0] + 1) * (vec2[:, 3] - vec2[:, 1] + 1)
        iou = i / (u - i + 1e-5)
        return iou

    @staticmethod
    def AP(vec1, vec2, threshold):
        matrix = Data_preprocess.get_iouMatrix(vec1, vec2) > threshold
        TP = np.sum(np.sum(matrix, axis=0) > 0)
        FP = np.sum(np.sum(matrix, axis=1) == 0)
        FN = np.sum(np.sum(matrix, axis=0) == 0)
        return [TP, FP, FN]

    @staticmethod
    def get_gt(gt):
        """
        :param gt: csv文件 一行单bbox的 目标检测标注
        :return: 转为字典形式{"imgname":bboxs(np.array)}
        """
        label = {}
        for idx in range(len(gt)):
            info = gt.iloc[idx]
            imgname = info.name
            if imgname not in label:
                label[imgname] = []
            label[imgname].append([info["x1"], info["y1"], info["x2"], info["y2"]])
        for key in label:
            label[key] = np.array(label[key])
        return label

    @staticmethod
    def detect_flip(img, bboxs, mirror=0.4, flip=0.4):
        bboxs = copy.copy((bboxs))
        h, w = img.shape[:2]
        if random.random() < mirror:
            img = cv2.flip(img, 1)
            bboxs[:, [0, 2]] = w - bboxs[:, [2, 0]]
        if random.random() < flip:
            img = cv2.flip(img, 0)
            bboxs[:, [1, 3]] = h - bboxs[:, [3, 1]]
        return img, bboxs

    @staticmethod
    def copy_past(img, label):
        """
        copy bbox 随机 past
        label x1,y1,x2,y2,cls
        """
        index = random.choices([i for i in range(label.shape[0])], k=3)
        h, w = img.shape[:2]
        h -= 1
        w -= 1
        for i in index:
            if (random.random() > 0.2): continue
            bbox = label[i]
            bh, bw = bbox[3] - bbox[1], bbox[2] - bbox[0]
            x1, y1 = random.randint(0, w - bw), random.randint(0, h - bh)
            x2, y2 = x1 + bw, y1 + bh
            new_label = np.array([[x1, y1, x2, y2, bbox[4]]])
            iou = Data_preprocess.get_iouMatrix(new_label, label)
            # print("iou \n", iou)
            if (np.sum(iou > 0.15)): continue
            img[y1:y2, x1:x2] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            label = np.vstack((label, new_label))
        return img, label

    @staticmethod
    def rotate(img, bbox, angel):
        #angel %= 90
        """
        输入img和bbox，角度。返回旋转后的img和bbox
        :param img:
        :param bbox:
        :param angel:
        :return:
        """
        degree = angel * np.pi/180
        height, width = img.shape[:2]
        # heightNew=int(width*np.abs(np.sin(angel))+height*np.abs(np.cos(angel)))
        # widthNew=int(height*np.abs(np.sin(angel))+width*np.abs(np.cos(angel)))
        heightNew = int(width * np.sin(degree) + height * np.cos(degree))
        widthNew = int(height * np.sin(degree) + width * np.cos(degree))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angel, 1)

        matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
        matRotation[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        newlabel = np.zeros(shape=(bbox.shape[0], 4, 2))
        newlabel[:, [0, 3], 0] = bbox[:, 0].reshape(-1, 1)
        newlabel[:, [0, 1], 1] = bbox[:, 1].reshape(-1, 1)
        newlabel[:, [1, 2], 0] = bbox[:, 2].reshape(-1, 1)
        newlabel[:, [2, 3], 1] = bbox[:, 3].reshape(-1, 1)
        newlabel = np.dot(newlabel.reshape(-1, 2), matRotation[:, :2].T) + matRotation[:, 2]
        newlabel = newlabel.reshape(-1, 4, 2)
        bbox[:, [0, 1, 2, 3]] = newlabel[:, [0, 1, 2, 3], [0, 1, 0, 1]]
        wh = bbox[:, 2:4] - bbox[:, :2]
        diff = wh * (45 - np.abs(angel - 45)) * 0.001
        diff = diff.astype(np.int)
        bbox[:, [0, 1]] += diff
        bbox[:, [2, 3]] -= diff
        return imgRotation, bbox

    @staticmethod
    def focal_loss(pred, label, is_valid=None, eps=1e-7):
        """
        :param is_valid: bool_列表, 即忽略掉anchor与gt的iou介于正负例阈值之间的
        focal 次方默认为 2
        :return:
        """
        #neg_valid = (1 - label) * (pred < 0.85).float()
        #CE_loss_neg = -1 * neg_valid * torch.log(1 + eps - pred)
        #neg_index = torch.sort(neg_valid, descending=True).indices < 5 * torch.sum(label).item()
        CE_loss_pos = -1 * (label * torch.log(pred + eps))
        CE_loss_neg = -1 * (1 - label) * torch.log(1 + eps - pred)
        neg_index = torch.sort(pred * (1 - label) * (pred < 0.9).float(), descending=True).indices < 6 * torch.sum(label).item()
        CE_loss = CE_loss_pos + CE_loss_neg
        focal_weight = (pred - label) ** 2  # torch.pow((pred - label), torch.tensor(2.0, dtype=torch.float, device=torch.device('cuda:0')))
        loss = CE_loss * focal_weight
        if is_valid is not None:
            loss = loss[is_valid * (neg_index + (label > 0))]
        else:
            loss = loss[(neg_index + (label > 0))]
        if (loss.size(0) == 0):return torch.tensor(0,device="cuda:0", dtype=torch.float)
        return torch.mean(loss)

    @staticmethod
    def smooth_L1(pred, label, is_valid=None, point=0.3):
        """
        :param point:  交界点
        """
        diff = torch.abs(pred - label)
        loss = torch.where(diff < point, diff * diff / (point * 2), diff)
        if is_valid is not None:
            loss = loss[is_valid]
        if (loss.size(0) == 0): return torch.tensor(0,device="cuda:0", dtype=torch.float)
        return torch.mean(loss)

if __name__ == "__main__":
    print(np.tan(45))