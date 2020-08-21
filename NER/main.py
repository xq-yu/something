import os
os.chdir('C:/Users/yuxiaoqiang/Desktop/coding/something/NER/')
#os.chdir('/content/drive/My Drive/colab_share/TextClassification')

import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer, AdamW
import torch.nn.functional as F
import torch.nn as nn
import torch
import nnmodel
import random
from sklearn.model_selection import train_test_split


f = open('C:/Users/yuxiaoqiang/Desktop/coding/something/NER/data/train.txt')
data = np.array(f.readlines())
f.close

split_points = [-1]+list(np.where(data=='\n'))
data = [x[0:-1].split(' ') for x in data]


for i,v in range(len(data)):
    sdfs
def read_data(file):
    with f = open(file):
        sdfsdf

        