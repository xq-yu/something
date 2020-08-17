import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_data(file):
    '''
    purpose:
        读取数据并切分单词
    input:
        file: 文件路径
    output:
        df: 输出数据集
    '''
    df = pd.read_csv(file, sep='\t')
    df.Phrase = df.Phrase.str.split(' +')
    df.Phrase = df.Phrase.map(lambda x: ' '.join(x))
    return df


class tokenizer():
    def __init__(self, words_dict, padding, oobIndex, padIndex):
        self.words_dict = words_dict
        self.oobIndex = oobIndex
        self.padIndex = padIndex

    def sentence_map(self, s, oobIndex):
        s_new = s.split(' ')
        s_new = [self.words_dict[word]
                 if word in self.words_dict else oobIndex for word in s_new]
        return s_new

    def sentence_padding(self, s, seq_len, padIndex):
        if len(s) >= seq_len:
            s_new = s[0:seq_len]
        else:
            s_new = [padIndex] * (seq_len-len(s))+s
        return s_new

    def __call__(self, sentences, padding):
        '''
        purpose:
            对单词进行 index 映射
        input:
            sentences: 输入pandas Series, 每个元素为一个句子的分好词的list
            padding:是否进行填充
        output:
            df: 输出数据集 df.shape=(len(col),len)
        '''
        sentences_new = [self.sentence_map(
            x, self.oobIndex) for x in sentences]
        if padding:
            max_len = max([len(x) for x in sentences])
            sentences_new = [self.sentence_padding(
                x, max_len, self.padIndex) for x in sentences_new]
        return {'input_ids': sentences_new}


def batch_iter(sentences, tokenizer, ID, batch_size, sorted, label=None):
    '''
    purpose:
        建立mini batch list
    input:
        sentences: pandas series 建立完索引的句子
        tokenizer: 单词编码器,包含encode 方法,将单词编码成模型索引序号
        ID: 样本标示列
        label: pandas series 文本标签
        batch_size: 每个 batch 的样本数
        sorted: 是否根据句子长度进行排序，保持每个batch中的句子长度尽量接近
    output:
        batch_iter: batch 迭代器,每个元素为（原始ID,sentence,label）
    '''

    df = pd.DataFrame({'ID': ID,
                        'sentences': sentences.copy(),
                        'seq_len': [len(x) for x in sentences]})
    if label is not None:
        df['label'] = label.copy()

    # 对sentence 按照长度进行排序
    if sorted:
        df = df.sort_values('seq_len').reset_index(drop=True)
    batch_num = len(df)//batch_size
    batch_iter = []
    for i in range(batch_num):
        if i < batch_num-1:
            X = list(df.sentences[i * batch_size: (i + 1) * batch_size])
            ID = list(df.ID[i * batch_size: (i + 1) * batch_size])
            y = list(df.label[i * batch_size: (i + 1) *
                              batch_size]) if label is not None else None
        else:
            X = list(df.sentences[i * batch_size:])
            ID = list(df.ID[i * batch_size:])
            y = list(df.label[i * batch_size:]) if label is not None else None

        # print(list(X))
        X = tokenizer(X, padding=True)['input_ids']

        if y is not None:
            tmp = (ID, torch.LongTensor(X), torch.LongTensor(y))
        else:
            tmp = (ID, torch.LongTensor(X))
        batch_iter.append(tmp)
    return batch_iter
