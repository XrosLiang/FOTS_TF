# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import re

import sys
import math
import numpy as np

input_gt_dir = 'test_gt_1080p_v1011_sess'
input_pred_dir = sys.argv[1]

def get_path(data_dir):
    files = []
    for ext in ['txt']:
        files.extend(glob.glob(os.path.join(data_dir, '*.{}'.format(ext))))
    return files


def contain_eng(str0):
    return bool(re.search('[a-z]', str0))


def rm_1st_digit(str0):
    if contain_eng(str0) and str0[0].isdigit():
        return str0[1:]
    else:
        return str0


def inter_num(a, b):
    return len(list(set(a) & set(b)))


def load_label(label_file):
    labels = []
    with open(label_file, 'r', encoding="utf-8-sig") as f:
        for line in f.readlines():
            line = line.replace('\xef\xbb\bf', '')
            line = line.replace('\xe2\x80\x8d', '')
            line = line.replace('\uff1a', ':')
            line = line.strip()
            line = line.split(',')
            try:
                temp_label = line[8]
            except:
                temp_label = 'null'
            labels.append(temp_label)
    return labels


gt_list = np.array(get_path(input_gt_dir))

s = 0
i_4 = 0
i_6 = 0
i_7 = 0
for txt in gt_list:
    gt_labels = sorted(load_label(txt))

    pred_list = os.path.join(input_pred_dir, txt.split("/")[1])
    if not os.path.exists(pred_list):
        print('text file {} does not exists'.format(pred_list))
        continue
    else:
        pred_labels = sorted(load_label(pred_list))
        s = s + 1

    qtr_list = ['ot', 'secondqtr', '1rdqtr', '4thqtr', '3reqtr', '2ndquarter', 'fourthqtr', '2ot', 'firstqtr', '3edqtr',
                '3rdquarter', '1stoqtr', '4thquarter', 'th', '3ed', 'end2nd', '1stqtr', 'wndqtr', 'final', '20t', '2q',
                '2neqtr', 'endof1stquarter', '2ndqyarter', 'overtime', '3tdqtr', '4thatr', '1etqtr', 'thiedqtr',
                '1atqtr', 'secondqtr', '1stquarter', '2rdqtr', '3rdqtr', 'ndqtr', '4yhqtr', '2nsqtr', '4rh', '4st',
                '1stotr', 'end1st', '4tnqtr', '4t', 'endof3rdquarter', '2ndqtr', 'thqtr', 'final/ot', '1srqtr',
                '3rdqtr ', '1stqtr ', '4yh', 'end3rdquarter', '4rdqtr', '3rsqtr', 'fiestqtr', 'quarter', '2ndqt',
                '3rdqte', '3qtr', '2ngqtr', 'end3rd', 'qrdqtr', '1stqtp', 'firstqt', '3raqtr', '1st', '3thqtr', '3rqtr',
                'decondqtr', '1dst', '4q', '3rdqrt', '1st', '2nd', '3rd', '4th', 'first', 'second', 'third', 'fourth',
                'thirdqtr', '4rtqtr', '1ST', '2ND', '3RD', '4TH', '1STQTR', '2NDQTR', '3RDQTR', '4THQTR']

    gt_labels_eng = [i for i in gt_labels if contain_eng(i) == 0]
    pred_labels_eng = [i for i in pred_labels if contain_eng(i) == 0]

    gt_labels_qtr = [rm_1st_digit(i) for i in gt_labels if i not in qtr_list]
    pred_labels_qtr = [rm_1st_digit(i) for i in pred_labels if i not in qtr_list]

    if inter_num(gt_labels_eng, pred_labels_eng) >= len(gt_labels_eng):
        i_4 = i_4 + 1

    if inter_num(gt_labels_qtr, pred_labels_qtr) >= len(gt_labels_qtr):
        i_6 = i_6 + 1

print('比分和时间完全相同（小框*4）', i_4 / s)
print('队名/比分和时间完全相同（小框*6）', i_6 / s)
