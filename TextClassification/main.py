import functions as myfun
import os
from sklearn.model_selection import train_test_split
import random
import nnmodel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
os.chdir('/Users/yu/Desktop/coding/github/something/TextClassification')
# 读取数据
trainset = myfun.read_data('./train.tsv')
trainset, valset = train_test_split(trainset, test_size=0.3, random_state=7)

########################################################################################################################
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
                                      hidden_dim=30,
                                      n_layers=1,
                                      bidirectional=self.bidirectional,
                                      dropout=0.5)
        # 全连接与分类层
        if self.bidirectional:
            self.fc1 = torch.nn.Linear(
                bias=True, in_features=30*2, out_features=40)
        else:
            self.fc1 = torch.nn.Linear(
                bias=True, in_features=30, out_features=40)

        self.fc2 = nn.Linear(40, 40)
        self.classifier = nn.Linear(40, 5)

    def forward(self, text):
        embedded = self.net_glove(text)
        h_output = self.net_rnn(embedded)
        x = F.relu(self.fc1(h_output))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x

net = mynetwork()
# 创建batch 迭代器，采用glove 的tokenizer 进行编码
train_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                    label=trainset.Sentiment,
                                    ID=trainset.PhraseId,
                                    tokenizer=net.net_glove.tokenizer,
                                    batch_size=100,
                                    sorted=True)
val_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                  label=trainset.Sentiment,
                                  ID=trainset.PhraseId,
                                  tokenizer=net.net_glove.tokenizer,
                                  batch_size=100,
                                  sorted=True)
random.shuffle(train_batch_iter)

########################################################################################################################
#                                                        BERT 模型                                                         
########################################################################################################################

'''
class mynetwork():
  def __init__(self):
    self.net_bert = nnmodel.netBert('bert_file', './bert-base-uncased-vocab.txt')

    # 全连接与分类层
    self.fc1 = torch.nn.Linear(self.net_bert.config_class().hidden_size,30)
    self.fc2 = torch.nn.Linear(30,30)
    self.classifier = torch.nn.Linear(30,5)

  def forward(self,text):
    last_hidden_state = self.net_bert(text)
    pooled_output = last_hidden_state[:,0,:] #取CLS对应输出(batch_size, hidden_size)
    x = F.relu(self.fc1(pooled_output))
    x = F.relu(self.fc2(x))
    x = self.classifier(x)
    return x
net = mynetwork()

train_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                    label=trainset.Sentiment,
                                    ID=trainset.PhraseId,
                                    tokenizer=net.net_bert.tokenizer,
                                    batch_size=100,
                                    sorted=True)
val_batch_iter = myfun.batch_iter(sentences=trainset.Phrase,
                                  label=trainset.Sentiment,
                                  ID=trainset.PhraseId,
                                  tokenizer=net.net_bert.tokenizer,
                                  batch_size=100,
                                  sorted=True)
random.shuffle(train_batch_iter)
'''
########################################################################################################################
#                                                       Fine Tune                                                        
########################################################################################################################
# 评估函数
def evaluation(model, dataset_iter):
    with torch.no_grad():
        num = 0
        corr = 0
        for batch in dataset_iter:
            X = batch[1]
            y = batch[2]
            y_hat = torch.argmax(model(X), dim=1)
            num = num+len(y)
            corr = corr+sum(y_hat == y)
    return corr.item()/num


# 7. 定义优化器
opitmizer = torch.optim.SGD(net.parameters(), lr=0.05)

# 8. 定义损失函数
lossfun = nn.CrossEntropyLoss()

# 9. epoch number
epoches = 1000

# 10. 定义模型
model = nnmodel.NN_Model(net=net,
                         optimizer=opitmizer,
                         lossfun=lossfun,
                         evalfun=evaluation)

# 12. 模型训练
model.train(train_iter=train_batch_iter,
            val_iter=val_batch_iter,
            epoch_num=1000,
            early_stop_rounds=3,
            print_freq=100)



  
