#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'get custom data'

import pandas as pd
import os
import sys
import math
import json
import numpy as np
import torch
import csv


def get_custom_labels_imgpathsfile():
    def xyxy2xywh(x):
        x = list(map(float, x))
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[0] = (x[0] + x[2]) / 2
        y[1] = (x[1] + x[3]) / 2
        y[2] = x[2] - x[0]
        y[3] = x[3] - x[1]
        # Normalize coordinates 0 - 1
        y[0] = round(y[0] / w,4)# width #大图片的宽
        y[2] = round(y[2] / w,4)
        y[1] = round(y[1] / h,4) # height
        y[3] = round(y[3] / h,4)
        return y

    img_dir = './training_img_1080p_v1011_comb/'
    label_dir = './training_gt_1080p_v1011_comb_v2/'
    csv_path = 'train_v1011.csv'
    dataset_str = " "

    w, h = 496, 280

    df = pd.read_csv(csv_path, encoding='utf-8', sep=',', header=None, names=['img', 'gt','xy'])
    #df['img_name'] = df['img'].apply(lambda x: x.split("\\")[-1].split(".")[0])
    df['img_name'] = df['img']
    #df['img_path'] = df['img'].apply(lambda x: img_dir + x.split("\\")[-1])
    df['img_path'] = df['img'].apply(lambda x: img_dir + x)
    df['labels'] = df['gt'].apply(lambda x: str(x).split("_"))

    for index,lines in enumerate(df['xy']):
        vertices = []
        labels = []
        points = list(map(int, lines.strip().split("_")))
        if len(points) % 4 != 0:
            print(lines)
        x1, y1, x2, y2 = points[:4]
        length = len(points)
        for i in range(4, length, 4):
            vertices.append(
                [points[i + 0], points[i + 1], points[i + 2], points[i + 1],
                 points[i + 2], points[i + 3],points[i + 0],points[i + 3]])
        for j in df['labels'][index]:
            labels.append(j)
        df_gt = pd.DataFrame(list(zip(vertices, labels)))
        df_gt['xy'] = df_gt[0]
        df_gt['gt'] = df_gt[1]
        df_gt = pd.concat([ pd.DataFrame(df_gt['xy'].values.tolist(), columns=['xy_' + str(x) for x in range(len(df_gt.loc[0,'xy']))]), df_gt['gt']], axis=1)

        vertices_comb = []
        if len(df_gt)<=5:
            df_gt.to_csv(label_dir + df['img_name'][index] + '.txt', index=0, header=0)
        elif len(df_gt)<=7:

            x1_0 = df_gt.iloc[:2]['xy_0'].min()
            y1_0 = df_gt.iloc[:2]['xy_1'].min()
            x1_1 = df_gt.iloc[:2]['xy_4'].max()
            y1_1 = df_gt.iloc[:2]['xy_5'].max()
# TODO: 标注时，应该以左上角xy坐标志之和最小；
            x1_gt = df_gt.iloc[:2]['gt'][(df_gt.iloc[:2]['xy_0']+df_gt.iloc[:2]['xy_1']) == (df_gt.iloc[:2]['xy_0']+df_gt.iloc[:2]['xy_1']).min()].values + '_' + df_gt.iloc[:2]['gt'][(df_gt.iloc[:2]['xy_0']+df_gt.iloc[:2]['xy_1']) == (df_gt.iloc[:2]['xy_0']+df_gt.iloc[:2]['xy_1']).max()].values

            x2_0 = df_gt.iloc[2:4]['xy_0'].min()
            y2_0 = df_gt.iloc[2:4]['xy_1'].min()
            x2_1 = df_gt.iloc[2:4]['xy_4'].max()
            y2_1 = df_gt.iloc[2:4]['xy_5'].max()
            #x2_gt = df_gt.iloc[2:4]['gt'][df_gt.iloc[2:4]['xy_0'] == df_gt.iloc[2:4]['xy_0'].min()].values + '_' + df_gt.iloc[2:4]['gt'][df_gt.iloc[2:4]['xy_0'] == df_gt.iloc[2:4]['xy_0'].max()].values
            x2_gt = df_gt.iloc[2:4]['gt'][(df_gt.iloc[2:4]['xy_0'] + df_gt.iloc[2:4]['xy_1']) == (df_gt.iloc[2:4]['xy_0'] + df_gt.iloc[2:4]['xy_1']).min()].values + '_' + df_gt.iloc[2:4]['gt'][(df_gt.iloc[2:4]['xy_0'] + df_gt.iloc[2:4]['xy_1']) == (df_gt.iloc[2:4]['xy_0'] + df_gt.iloc[2:4]['xy_1']).max()].values


            vertices_comb.append([x1_0,])
            #print([x1_0,y1_0,x1_1,y1_0,x1_1,y1_1,x1_0,y1_1,x1_gt[0]])
            #print([x2_0,y2_0,x2_1,y2_0,x2_1,y2_1,x2_0,y2_1,x2_gt[0]])
            try:
                df_gt_comb = pd.DataFrame([[x1_0,y1_0,x1_1,y1_0,x1_1,y1_1,x1_0,y1_1,x1_gt[0]],
                                           [x2_0,y2_0,x2_1,y2_0,x2_1,y2_1,x2_0,y2_1,x2_gt[0]],
                                            df_gt.iloc[4].values,df_gt.iloc[5].values,df_gt.iloc[6].values
                                          ])
            except:
                df_gt_comb = pd.DataFrame([[x1_0, y1_0, x1_1, y1_0, x1_1, y1_1, x1_0, y1_1, x1_gt[0]],
                                           [x2_0, y2_0, x2_1, y2_0, x2_1, y2_1, x2_0, y2_1, x2_gt[0]],
                                           df_gt.iloc[4].values, df_gt.iloc[5].values
                                           ])

            df_gt_comb.to_csv(label_dir + df['img_name'][index] + '.txt',index=0,header=0)


if __name__ == '__main__':
    get_custom_labels_imgpathsfile()
