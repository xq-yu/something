import os
os.chdir('C:/Users/yuxiaoqiang/Desktop/coding/something/TextClassification/')
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

import importlib
importlib.reload(nnmodel)
import functions as myfun

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
trainset = myfun.read_data('./train.tsv')
trainset, valset = train_test_split(trainset, test_size=0.3, random_state=7)

testset = myfun.read_data('./test.tsv')

########################################################################################################################
#                                                              1.
#                                                     RNN 架构文本多分类模型
#                                                      一层RNN+3层全连接
#                                                   glove.6B.50d词向量嵌入
########################################################################################################################
# 定义network
class mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_glove = nnmodel.nnGlove(
            glove_file='./glove.6B.50d.txt', oobIndex=0, padIndex=0)
        self.bidirectional = False
        self.net_rnn = nnmodel.netRNN(rnn_cell='LSTM',
                                      input_size=self.net_glove.embedMatrix.shape[1],
                                      hidden_dim=10,
                                      n_layers=1,
                                      bidirectional=self.bidirectional,
                                      dropout=0.5)
        # 全连接与分类层
        # 文本分类全连接层
        if self.bidirectional:
            in_features = 10*2
        else:
            in_features = 10
        self.fc1 = torch.nn.Linear(bias=True, in_features=in_features, out_features=40)
        self.fc2 = nn.Linear(40, 10)
        self.classifier = nn.Linear(10, 5)

    def forward(self, text):
        embedded = self.net_glove(text)
        h_n = self.net_rnn(embedded)[:,-1,:]   # 选取最后一个 step 的输出
        x = F.relu(self.fc1(h_n))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x

net = mynetwork().to(device)
# 创建batch 迭代器，采用glove 的tokenizer 进行编码
train_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                    label=trainset.Sentiment,
                                    ID=trainset.PhraseId,
                                    tokenizer=net.net_glove.tokenizer,
                                    batch_size=100,
                                    sorted=True)
val_batch_iter = myfun.batch_iter(sentences=valset.Phrase,
                                  label=valset.Sentiment,
                                  ID=valset.PhraseId,
                                  tokenizer=net.net_glove.tokenizer,
                                  batch_size=100,
                                  sorted=True)
test_batch_iter = myfun.batch_iter(sentences=testset.Phrase,
                                   label=None,
                                   ID=testset.PhraseId,
                                   tokenizer=net.net_glove.tokenizer,
                                   batch_size=100,
                                   sorted=True)

random.shuffle(train_batch_iter)


########################################################################################################################
#                                                              2.
#                                                     RNN 架构文本多分类模型
#                                                      一层RNN+3层全连接
#                                                        增加attention层
#                                                   glove.6B.50d词向量嵌入
########################################################################################################################
'''
# 定义network

class mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_glove = nnmodel.nnGlove(
            glove_file='./glove.6B.50d.txt', oobIndex=0, padIndex=0)
        self.bidirectional = False
        self.net_rnn = nnmodel.netRNN(rnn_cell='LSTM',
                                      input_size=self.net_glove.embedMatrix.shape[1],
                                      hidden_dim=40,
                                      n_layers=1,
                                      bidirectional=self.bidirectional,
                                      dropout=0.5)
        # 全连接与分类层
        if self.bidirectional:
            self.fc1 = torch.nn.Linear(bias=True,
                                       in_features=40*2,
                                       out_features=40)
        else:
            self.fc1 = torch.nn.Linear(bias=True,
                                       in_features=40,
                                       out_features=40)

        self.fc2 = torch.nn.Linear(bias=True,
                                   in_features=40,
                                   out_features=40)
        self.net_attenion = nnmodel.netAttention()
        self.classifier = nn.Linear(40, 5)

    def forward(self, text):
        embedded = self.net_glove(text)
        h_output = self.net_rnn(embedded)  # 最后一层隐藏层的输出序列,作为attention key和value
        h_n = h_output[:, -1:, :]  # 最后一个step的隐藏层输出

        x = F.relu(self.fc1(h_n))  # 做一层全连接，作为 query_seqß
        x = F.relu(self.fc2(x))
        x = self.net_attenion(key_seq=h_output,
                              value_seq=h_output,
                              query_seq=x)
        x = self.classifier(x[:, 0, :])
        return x


net = mynetwork().to(device)
# 创建batch 迭代器，采用glove 的tokenizer 进行编码
train_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                    label=trainset.Sentiment,
                                    ID=trainset.PhraseId,
                                    tokenizer=net.net_glove.tokenizer,
                                    batch_size=100,
                                    sorted=True)
val_batch_iter = myfun.batch_iter(sentences=valset.Phrase,
                                  label=valset.Sentiment,
                                  ID=valset.PhraseId,
                                  tokenizer=net.net_glove.tokenizer,
                                  batch_size=100,
                                  sorted=True)
random.shuffle(train_batch_iter)
'''
########################################################################################################################
#                                                        BERT 模型
########################################################################################################################

'''
class mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_bert = nnmodel.netBert(
            './bert-base-uncase/', './bert-base-uncase/bert-base-uncased-vocab.txt')

        # 全连接与分类层
        self.fc1 = torch.nn.Linear(
            self.net_bert.bert.config_class().hidden_size, 30)
        self.fc2 = torch.nn.Linear(30, 30)
        self.classifier = torch.nn.Linear(30, 5)

        # dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, text):
        # 取CLS对应输出(batch_size, hidden_size)
        last_hidden_state = self.net_bert(text)
        x = last_hidden_state[:, 0, :]
        #x = self.dropout(pooled_output)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.classifier(x)
        return x


net = mynetwork().to(device)

train_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                    label=trainset.Sentiment,
                                    ID=trainset.PhraseId,
                                    tokenizer=net.net_bert.tokenizer,
                                    batch_size=100,
                                    sorted=True)
val_batch_iter = myfun.batch_iter(sentences=valset.Phrase,
                                  label=valset.Sentiment,
                                  ID=valset.PhraseId,
                                  tokenizer=net.net_bert.tokenizer,
                                  batch_size=100,
                                  sorted=True)
test_batch_iter = myfun.batch_iter(sentences=testset.Phrase,
                                   label=None,
                                   ID=testset.PhraseId,
                                   tokenizer=net.net_bert.tokenizer,
                                   batch_size=100,
                                   sorted=True)

random.shuffle(train_batch_iter)
'''
########################################################################################################################
#                                                       Fine Tune
########################################################################################################################
# 评估函数
def evaluation(y_hat,y):
    '''
    purpose:
        指标评估函数, 监测损失函数或者其他业务指标
    input:
        y_hat: np.array 模型预测结果，与nnmdoel.NN_model.predict 输出一致
        y: np.array 真实标签
    output:
        自定义指标
    '''
    y_hat = np.argmax(y_hat, axis=1)
    return sum(y_hat == y) / len(y)


# 7. 定义优化器
#opitmizer = torch.optim.SGD(net.parameters(), lr=0.1)
# opitmizer = AdamW(net.parameters(),
#                   lr=0.000001,
#                   betas=(0.9, 0.999),
#                   eps=1e-06,
#                   weight_decay=0.0,
#                   correct_bias=True)

opitmizer = AdamW(net.parameters(),
                  lr=0.0001,
                  betas=(0.9, 0.999),
                  eps=1e-06,
                  weight_decay=0.0,
                  correct_bias=True)

# 8. 定义损失函数
lossfun = nn.CrossEntropyLoss()

# 9. epoch number
epoches = 1000

# 10. 定义模型
model = nnmodel.NN_Model(net=net,
                         device=device,
                         optimizer=opitmizer,
                         lossfun=lossfun,
                         evalfun=evaluation,
                         track_params=[])#[net.fc1.weight[1][1],net.net_rnn.rnn.weight_hh_l0[1][1],net.net_rnn.rnn.weight_hh_l0[3][1]])

# 12. 模型训练
model.train(train_iter=train_batch_iter,
            val_iter=val_batch_iter,
            epoch_num=1000,
            early_stop_rounds=3,
            print_freq=100)

########################################################################################################################
#                                                       Result Submit
########################################################################################################################
def submit(model, testset_iter):
    ID, y_hat = model.predict(testset_iter)
    y_hat = np.argmax(y_hat, axis=1)
    df = pd.DataFrame({'PhraseId': ID, 'Sentiment': y_hat})
    # df.to_csv('./submisstion.csv',index = False)
    return df

df = submit(model, test_batch_iter)
df = df.sort_values('PhraseId').reset_index(drop=True)
df.to_csv('./submisstion.csv', index=False)


