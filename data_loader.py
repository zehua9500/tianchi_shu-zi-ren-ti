import torch
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
from Data_PreProcess import Data_preprocess as dp

def get_gt():
    gt_fold = r"E:\cancer\imglabel\groundTruth"
    gt_list = os.listdir(gt_fold)
    gt = pd.DataFrame(columns={"x1","y1","x2","y2"})
    gt.index.name = "filename"
    for t in gt_list:
        gt = gt.append(pd.read_csv(os.path.join(gt_fold, t), index_col=0))
    gt = gt.reindex(columns=["x1","y1","x2","y2"])
    return pd.DataFrame(gt, dtype='int')


def collate(batch):
    # print(len(batch[0]))
    # print("batch[2] ",batch[2])
    imgbatch = np.stack([b[0] for b in batch], 0)
    imgbatch = torch.from_numpy(imgbatch).cuda().float()
    imgbatch = imgbatch.permute((0, 3, 1, 2))

    labelnums = max([b[2] for b in batch])
    batch_size = len(batch)
    Labels = np.ones((batch_size, labelnums, 5)) * -1
    for idx, b in enumerate(batch):
        length = b[1].shape[0]
        Labels[idx, :length] = b[1]
    return imgbatch, Labels #Label需为numpy类型，去匹配gt


class DataGenerate(Dataset):
    def __init__(self, batch_size=2, input_size=800, gt = None, is_training = True):
        if(gt is None):self.gt = get_gt()
        else:self.gt = gt
        self.train_path = r"E:\cancer\train_data"
        self.imglist = list(set(self.gt.index))  # os.listdir(self.train_path)
        self.input_size = input_size
        self.batch_size = batch_size
        self.count = 0
        self.singleCount = 0
        self.is_training = is_training

    def __len__(self):
        random.shuffle(self.imglist)
        #print("shuffle list, list[0] is ", self.imglist[0])
        self.count = 0
        print("trainning set = ", len(self.gt))
        return len(self.gt)  # - len(self.gt)%self.batch_size

    def split(self):
        img = cv2.imread(os.path.join(self.train_path, self.imglist[self.count]))
        label = self.gt.loc[self.imglist[self.count]]
        label = np.array(label).reshape(-1, 4)  # .astype(np.int)
        self.count += 1

        angel = random.randint(0,90)
        if self.is_training:
            img, label = dp.rotate(img, label, angel)
            label = np.clip(label, 1, None)
        h, w = img.shape[:2]
        h_limit = h - self.input_size
        w_limit = w - self.input_size
        self.singleCount = len(label)
        self.imgbatch = []
        self.labelbatch = []

        for i in range(len(label)):
            info = label[i]
            x1, y1, x2, y2 = info[0], info[1], info[2], info[3]
            index_x = random.randint(max(0, x2 - self.input_size), min(w_limit, x1 - 1))
            index_y = random.randint(max(0, y2 - self.input_size), min(h_limit, y1 - 1))
            self.imgbatch.append(img[index_y:index_y + self.input_size, index_x:index_x + self.input_size])

            tempLabel = np.array(label)
            tempLabel[:, [0, 2]] -= index_x
            tempLabel[:, [1, 3]] -= index_y
            tempLabel = np.clip(tempLabel,0,self.input_size)
            idx = ((tempLabel[:,2] - tempLabel[:,0]) * (tempLabel[:,3] - tempLabel[:,1])) > 100
            cls = np.ones(shape = (np.sum(idx),1),dtype=np.int)
            self.labelbatch.append(np.hstack((tempLabel[idx], cls))) #label为 x1,y1,x2,y2 numpy类型


    def __getitem__(self, index):
        """
        单张图太大，切成小图不好随机取。  再改改
        """
        if (self.singleCount == 0): self.split()
        self.singleCount -= 1
        label = self.labelbatch[self.singleCount]
        img = self.imgbatch[self.singleCount]
        if self.is_training:
            img,label = dp.detect_flip(img,label)
            img, label = dp.copy_past(img,label)
            #if(random.random() < 0.5):
            #    img, label = dp.rotate(img,label)
        return img, label, label.shape[0]