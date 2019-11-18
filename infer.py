import kfbReader as kr
import torch
import numpy as np
import os
import time
import json
from Net3 import Net
from torch.utils.data import Dataset, DataLoader


def collate_sub(batch):
    img = torch.from_numpy(batch[0][0].transpose(2, 0, 1)[np.newaxis,]).cuda().float()
    return img, batch[0][1], batch[0][2]


class SubGenerate(Dataset):
    def __init__(self, imgpath, input_size):
        self.imgpath = imgpath
        self.input_size = input_size
        self.reader = kr.reader()
        self.reader.ReadInfo(imgpath, 20, True)
        self.Width = self.reader.getWidth()
        self.Height = self.reader.getHeight()
        self.stride = int(input_size * 0.85)
        self.index_w -= self.stride
        self.index_h = 0

    def __len__(self):
        length = ((self.Width - self.input_size) // self.stride + 1) * ((self.Height - self.input_size) // self.stride + 1)
        self.Width -= self.input_size
        self.Height -= self.input_size
        # print("len = ", length)
        return length

    def __getitem__(self, index):
        self.index_w += self.stride
        if (self.index_w + self.input_size > self.Width):
            self.index_w = 0
            self.index_h += self.stride
        img = self.reader.ReadRoi(self.index_w, self.index_h, self.input_size, self.input_size, 20)
        return img, self.index_w, self.index_h


def collate(batch):
    return batch[0][0], batch[0][1]


class InferGenerate(Dataset):
    def __init__(self, data_path, input_size):
        self.data_path = data_path
        self.datalist = os.listdir(data_path)
        self.input_size = input_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data_name = self.datalist[index]
        data_path = os.path.join(self.data_path, data_name)
        sub_generater = DataLoader(SubGenerate(imgpath=data_path, input_size=self.input_size), \
                                   batch_size=1, drop_last=False, collate_fn=collate_sub)
        return sub_generater, data_name


data_path = r"E:\cancer\test_0"
input_size = 864

# 2阶段 预测
model = Net(input_size=input_size, batch_size=1, is_training=False)
model.load_state_dict(torch.load(r"E:\cancer\model\model_include3\eps=3.t7"), strict=False)
#model = model.cuda()
model.eval()
model = model.cuda()
infer_generate = DataLoader(InferGenerate(data_path=data_path, input_size=input_size), \
                            batch_size=1, drop_last=False, collate_fn=collate)
with torch.no_grad():
    for idx, dat in enumerate(infer_generate):
        dat_name = dat[1]
        js = []
        start = time.time()
        for img, w, h in dat[0]:
            _, s2_res = model(img)
            del img
            if(s2_res.size(0) == 0):continue
            s2_res[:, 2:4] -= s2_res[:, :2]
            s2_res[:, 0] += w
            s2_res[:, 1] += h
            bboxs = s2_res.detach().cpu().numpy().astype(np.float)
            for th in range(bboxs.shape[0]):
                bbox = bboxs[th]
                js.append({"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3], "p": bbox[4]})
            #print("total time : %.5f", time.time() - start)
        with open(r"D:\ans\%s.json" % dat_name[:-4], 'w') as f:
            json.dump(js, f)
        print("No%d process, time = %f" % (idx, time.time() - start))