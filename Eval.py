import torch
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Data_PreProcess import Data_preprocess as dp
import kfbReader as kr
#from Net import RetinaNet
from Generate_Anchor import generate_prior_anchor, get_prior_anchor
import pandas as pd
import time


def eval(model, thres, input_size, s2_threshold=0.5):
    model.eval()
    data_path = r"E:\cancer\train_data"
    gt = pd.read_csv(r"E:\cancer\imglabel\val\3_gt.csv", index_col=0)
    gt = dp.get_gt(gt)
    datalist = list(gt.keys())
    #input_size = 1280
    TP, FP, FN = [], [], []
    with torch.no_grad():
        start = time.time()
        for idx, dat_name in enumerate(datalist):
            #print(idx)
            src_img = cv2.imread(os.path.join(data_path, dat_name))
            height, width = src_img.shape[:2]
            ans = []
            for h in range(0, height, int(input_size * 0.8)):  # 后续改成 d = self.input_size/2
                h = h if h + input_size < height else height - input_size

                for w in range(0, width, int(input_size * 0.8)):
                    w = w if w + input_size < width else width - input_size
                    # img = reader.ReadRoi(w,h,input_size,input_size,20)
                    img = src_img[h:h + input_size, w:w + input_size]
                    img = img.transpose(2, 0, 1)[np.newaxis,]
                    img = torch.from_numpy(img).cuda().float()

                    pred = model(img, threshold=thres,s2_threshold=s2_threshold)
                    pred = pred.detach().cpu().numpy()
                    pred[:, [0, 2]] += w
                    pred[:, [1, 3]] += h
                    index_nms = dp.py_cpu_nms(pred)
                    ans.append(pred[index_nms])
            ans = np.vstack(ans)
            ans[:, [0, 2]] = np.clip(ans[:, [0, 2]], 0, width)
            ans[:, [1, 3]] = np.clip(ans[:, [1, 3]], 0, width)

            index_nms = dp.py_cpu_nms(ans)
            ans = ans[index_nms]
            label = gt[dat_name]
            tp, fp, fn = dp.AP(ans, label, threshold=0.5)
            TP.append(tp)
            FP.append(fp)
            FN.append(fn)

    TP = np.array(TP)
    FP = np.array(FP)
    FN = np.array(FN)
    model.train(True)
    precision = TP / (TP + FP + 1e-5)
    print("TP = %d | FP = %d | FN = %d | precision = %.5f"%(np.sum(TP), np.sum(FP), np.sum(FN), np.mean(precision)))
    # print("recall \n", TP / (TP + FN + 1e-5))
    print("time = %.3f" % (time.time() - start))
