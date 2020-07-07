import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel,BertTokenizer
import random
import functions as myfun
import pandas as pd
import csv

class nnGlove(nn.Module):
    '''
    purpose:
        导入glove模型，建立基于glove 的embedding嵌入层
    
    '''
    def __init__(self,glove_file,oobIndex,padIndex):
        super().__init__()
        self.embedMatrix,self.words_dict = self.load_glove(glove_file)
        self.oobIndex = oobIndex
        self.padIndex = padIndex
        self.tokenizer = self.tokenizerBuiler()
        self.embedding_layer = nn.Embedding(self.embedMatrix.shape[0],self.embedMatrix.shape[1])
        self.__loadWeight__()

    def __loadWeight__(self):
        self.embedding_layer.weight.data.copy_(torch.tensor(self.embedMatrix))

    def tokenizerBuiler(self):
        tokenizer = myfun.tokenizer(words_dict = self.words_dict,
                                    padding=True,
                                    oobIndex = self.oobIndex,
                                    padIndex = self.padIndex)
        return tokenizer

    def load_glove(self,filename):
        '''
        purpose:
            导入 glove 词向量和单词索引关系
        input:
            file: glove词向量文件
        output:
            mebedding: embedding 矩阵，第0行表示OOB单词向量
            words_dict: 单词-索引关系（1～N）
        note:
            oob 单词的索引为0，词向量也是全0向量
        '''   
        embedding = pd.read_csv(filename,sep = ' ',quoting=csv.QUOTE_NONE,header=None)
        embedding = embedding.drop_duplicates(subset = [0])
        # 建立单词映射字典，从1～N
        words_dict = {embedding.iloc[value,0]:value+1 for value in range(embedding.shape[0])}
        # embedding 矩阵，第0行为OOB单词向量
        embedding = embedding.iloc[:,1:].values
        embedding = np.row_stack([[0]*embedding.shape[1],embedding])
        return embedding,words_dict

    def forward(self,text):
        embedded = self.embedding_layer(text)
        return embedded



class netRNN(nn.Module):
    '''
    purpose:
        建立 RNN 框架
    input:
        rnn_cell: RNN 单元类型（'LSTM','GRU'）
        vocab_size: 词袋/embedding矩阵大小
        embedding_size: 词向量长度
        hidden_dim: RNN 隐藏单元数量
        n_layers: RNN隐藏个数
        bidirectional: 是否双向 ('True','False')
        dropout: RNN dropout 比例
    note:
        单向RNN取最后一个时间步的隐藏层输出
        双向RNN将两端的输出进行拼接
    '''
    def __init__(self, rnn_cell , input_size, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.rnn_cell = rnn_cell
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # RNN layer
        if self.rnn_cell=='GRU':
            self.rnn = nn.GRU(input_size=input_size, 
                              hidden_size=hidden_dim,
                              num_layers=n_layers, 
                              bidirectional=bidirectional, 
                              dropout=dropout,
                              batch_first=True,
                              bias = True)        
        elif self.rnn_cell=='LSTM':
            self.rnn = nn.LSTM(bias=True,
                               input_size=input_size,
                               hidden_size=hidden_dim,
                               bidirectional = bidirectional,
                               batch_first = True,
                               num_layers = n_layers)
        #文本分类全连接层
        if bidirectional:
            self.fc1 = torch.nn.Linear(bias=True,in_features=hidden_dim*2,out_features=40)
        else:
            self.fc1 = torch.nn.Linear(bias=True,in_features=hidden_dim,out_features = 40)

        self.fc2   = nn.Linear(40, 40)       
        self.fc3   = nn.Linear(40, 5)     #定义fc4（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
    
    def forward(self, sequence):
        # sequence:[batch_size,sen_len, feature]
        if self.rnn_cell == 'GRU':
            output, h_n = self.rnn(sequence)      
            output = output.view(sequence.shape[0],          #batch
                                 sequence.shape[1],          #seq_len 
                                 self.num_directions,        #num_directions
                                 self.rnn.hidden_size)       #hidden_size
            h_n = h_n.view(self.rnn.num_layers,              #num_layers
                           self.num_directions,              #num_directions
                           sequence.shape[0],                #batch
                           self.rnn.hidden_size)             #hidden_size                   

        elif self.rnn_cell =='LSTM':
            output, (h_n, c_n) = self.rnn(sequence)
            output = output.view(sequence.shape[0],    #batch
                                 sequence.shape[1],    #seq_len                                 
                                 self.num_directions,  #num_directions
                                 self.rnn.hidden_size) #hidden_size
            h_n = h_n.view(self.rnn.num_layers,        #num_layers
                           self.num_directions,        #num_directions
                           sequence.shape[0],          #batch
                           self.rnn.hidden_size)       #hidden_size 
            c_n = c_n.view(self.rnn.num_layers,        #num_layers 
                           self.num_directions,        #num_directions
                           sequence.shape[0],          #batch
                           self.rnn.hidden_size)       #hidden_size
        
        ########################文本分类全连接层#########################
        if self.num_directions ==2:
            h_output = torch.cat([h_n[-1,-1,:,:],h_n[-1,-2,:,:]],dim = 1)
        else:
            h_output = h_n[-1,-1,:,:]

        return h_output

class netBert(nn.Module):
    '''
    purpose:
        建立 Bert 框架
    input:
        bert_file: bert 模型文件
        vocab_file: 单词索引文件
    note:

    '''
    def __init__(self,bert_file,vocab_file):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)

        self.bert = self.__bert_load__(bert_file) #载入bert模型
        self.dropout = nn.Dropout(self.bert_model.config_class().attention_probs_dropout_prob)  #继承基础bert模型的dropout

    def __bert_load__(self,bert_file):
        '''
        purpose:
            通过文件倒入bert 框架
        input:
            bert_file: bert 模型文件(your_dir/with/config.json/and/pytorch_model.bin)
        output:
            bert_model: nn.model        
        '''
        bert_model =  BertModel.from_pretrained(bert_file,
                                                output_hidden_states=True,
                                                output_attentions=True)
        return bert_model

    def forward(self, text):
        last_hidden_state = self.bert(text)[0]             #最后一层输出(batch_size, sequence_length, hidden_size)
        return last_hidden_state


class NN_Model:
    def __init__(self,net,optimizer=None,lossfun=None,evalfun=None):
        self.net = net
        self.optimizer = optimizer
        self.lossfun = lossfun
        self.evalfun = evalfun
        self.learning_curve = []
    
    def train(self,train_iter,val_iter,epoch_num,early_stop_rounds,print_freq):
        for epoch in range(epoch_num):
            loss_sum = 0  # 累计损失
            sample_count = 0 # 累计样本
            perform_batch = []
            for k,batch in enumerate(train_iter):
                X = batch[1]
                y = batch[2]
                y_hat = self.net(X)
                loss = self.lossfun(y_hat,y)

                # 反向传播梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 训练过程打印
                loss_sum += loss.item()
                sample_count += len(y)

                # minibatch 训练过程打印
                if k%print_freq==0:
                    perform_batch.append([k,loss_sum/sample_count])
                    plt.subplot(211)
                    plt.cla()
                    plt.plot([x[0] for x in perform_batch],[x[1] for x in perform_batch],'.-')
                    plt.xlabel('batch_number')
                    plt.ylabel('average loss in current epoch')
                    #plt.title('procession: %s of %s batches_total; %s epoch of %s epochs total'%(  k,len(train_iter),epoch,epoch_num ))
                    plt.grid()
                    plt.pause(0.3)
                    #print( 'procession: %s of %s batches_total; %s epoch of %s epochs total'%(  k,len(train_iter),epoch,epoch_num ) )
                    #print( 'average loss in current epoch: %s'%( loss_sum/sample_count ) )

            # epoch 结果打印
            self.learning_curve.append([epoch,self.evalfun(self.net,train_iter),self.evalfun(self.net,val_iter)])
            plt.subplot(212)
            plt.cla()

            plt.plot([x[0] for x in self.learning_curve],
                     [x[1] for x in self.learning_curve],
                     '.-',
                     label = 'train')
            plt.plot([x[0] for x in self.learning_curve],
                     [x[2] for x in self.learning_curve],
                     '.-',
                     label = 'validation' )
            
            plt.xlabel('epoch_num')
            plt.ylabel('evaliation criteria')
            #plt.title('epoch summary')
            plt.grid()
            plt.pause(0.3)
            # print('#################  epoch summary %s ################'%epoch)
            # print('total loss: %s '%loss_sum)
            # print('evaliation criteria:')
            # print('train: %s ; validation: %s ' %(self.learning_curve[-1][0],self.learning_curve[-1][1]))
            # print('####################################################')

    def predict(self,test_iter):
        with torch.no_grad():
            for batch in test_iter:
                X = batch[1]
                y_hat = self.net(X)
                if re:
                    re = np.row_stack((re,y_hat.numpy()))
                else:
                    re = y_hat.numpy()
        return re

    

        
