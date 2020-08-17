import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel, BertTokenizer, AdamW
import random
import functions as myfun
import pandas as pd
import csv


########################################################################################################################
#                                                        Glove 模块
########################################################################################################################
class nnGlove(nn.Module):
    '''
    purpose:
        导入glove模型，建立基于glove 的embedding嵌入层
    '''

    def __init__(self, glove_file, oobIndex, padIndex):
        super().__init__()
        self.embedMatrix, self.words_dict = self.load_glove(glove_file)
        self.oobIndex = oobIndex
        self.padIndex = padIndex
        self.tokenizer = self.tokenizerBuiler()
        self.embedding_layer = nn.Embedding(
            self.embedMatrix.shape[0], self.embedMatrix.shape[1])
        self.__loadWeight__()

    def __loadWeight__(self):
        self.embedding_layer.weight.data.copy_(torch.tensor(self.embedMatrix))

    def tokenizerBuiler(self):
        tokenizer = myfun.tokenizer(words_dict=self.words_dict,
                                    padding=True,
                                    oobIndex=self.oobIndex,
                                    padIndex=self.padIndex)
        return tokenizer

    def load_glove(self, filename):
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
        embedding = pd.read_csv(
            filename, sep=' ', quoting=csv.QUOTE_NONE, header=None)
        embedding = embedding.drop_duplicates(subset=[0])
        # 建立单词映射字典，从1～N
        words_dict = {embedding.iloc[value, 0]: value +
                      1 for value in range(embedding.shape[0])}

        # embedding 矩阵，第0行为OOB单词向量
        embedding = embedding.iloc[:, 1:].values
        embedding = np.row_stack([[0]*embedding.shape[1], embedding])
        return embedding, words_dict

    def forward(self, text):
        embedded = self.embedding_layer(text)
        return embedded


########################################################################################################################
#                                                        RNN 模块
########################################################################################################################
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
    output:
        output: 最后一层RNN 的输出 [batch_size, seq_len, hidden_size(*3)]
    note:
        单向RNN取最后一个时间步的隐藏层输出
        双向RNN将两端的输出进行拼接
    '''

    def __init__(self, rnn_cell, input_size, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.rnn_cell = rnn_cell
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # RNN layer
        if self.rnn_cell == 'GRU':
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              batch_first=True,
                              bias=True)
        elif self.rnn_cell == 'LSTM':
            self.rnn = nn.LSTM(bias=True,
                               input_size=input_size,
                               hidden_size=hidden_dim,
                               bidirectional=bidirectional,
                               batch_first=True,
                               num_layers=n_layers)

    def forward(self, sequence):
        # sequence:[batch_size,sen_len, feature]
        if self.rnn_cell == 'GRU':
            output, h_n = self.rnn(sequence)
            output = output.view(sequence.shape[0],  # batch
                                 sequence.shape[1],  # seq_len
                                 self.num_directions,  # num_directions
                                 self.rnn.hidden_size)  # hidden_size
            h_n = h_n.view(self.rnn.num_layers,  # num_layers
                           self.num_directions,  # num_directions
                           sequence.shape[0],  # batch
                           self.rnn.hidden_size)  # hidden_size

        elif self.rnn_cell == 'LSTM':
            output, (h_n, c_n) = self.rnn(sequence)
            output = output.view(sequence.shape[0],     # batch
                                 sequence.shape[1],     # seq_len
                                 self.num_directions,   # num_directions
                                 self.rnn.hidden_size)  # hidden_size
            h_n = h_n.view(self.rnn.num_layers,  # num_layers
                           self.num_directions,  # num_directions
                           sequence.shape[0],    # batch
                           self.rnn.hidden_size)  # hidden_size
            c_n = c_n.view(self.rnn.num_layers,  # num_layers
                           self.num_directions,  # num_directions
                           sequence.shape[0],  # batch
                           self.rnn.hidden_size)  # hidden_size

        ########################文本分类全连接层#########################
        if self.num_directions == 2:
            output = torch.cat([output[:, :, 0, :], output[:, :, 1, :]], dim=2)
            # h_output = torch.cat([h_n[-1, -1, :, :], h_n[-1, -2, :, :]], dim=1)
        else:
            # h_output = h_n[-1, -1, :, :]
            output = output[:, :, 0, :]

        return output


########################################################################################################################
#                                                        Bert 模块
########################################################################################################################
class netBert(nn.Module):
    '''
    purpose:
        建立 Bert 框架
    input:
        bert_file: bert 模型文件
        vocab_file: 单词索引文件
    note:

    '''

    def __init__(self, bert_file, vocab_file):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)

        self.bert = self.__bert_load__(bert_file)  # 载入bert模型
        self.dropout = nn.Dropout(self.bert.config_class(
        ).attention_probs_dropout_prob)  # 继承基础bert模型的dropout

    def __bert_load__(self, bert_file):
        '''
        purpose:
            通过文件倒入bert 框架
        input:
            bert_file: bert 模型文件(your_dir/with/config.json/and/pytorch_model.bin)
        output:
            bert_model: nn.model        
        '''
        bert_model = BertModel.from_pretrained(bert_file,
                                               output_hidden_states=True,
                                               output_attentions=True)
        return bert_model

    def forward(self, text):
        # 最后一层隐藏单元输出(batch_size, sequence_length, hidden_size)
        last_hidden_state = self.bert(text)[0]
        return last_hidden_state


########################################################################################################################
#                                                        模型训练预测集成
########################################################################################################################
class NN_Model:
    def __init__(self, net, device, optimizer=None, lossfun=None, evalfun=None):
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.lossfun = lossfun
        self.evalfun = evalfun
        self.learning_curve = []

    def train(self, train_iter, val_iter, epoch_num, early_stop_rounds, print_freq):
        for epoch in range(epoch_num):
            loss_sum = 0  # 累计损失
            sample_count = 0  # 累计样本
            perform_batch = []
            for k, batch in enumerate(train_iter):
                self.net.train()   # 将模型设置成训练模式
                X = batch[1].to(self.device)
                y = batch[2].to(self.device)
                y_hat = self.net(X)
                loss = self.lossfun(y_hat, y)

                # 反向传播梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 训练过程打印
                loss_sum += loss.item()
                sample_count += len(y)

                # minibatch 训练过程打印
                if k % print_freq == 0:
                    perform_batch.append([k, loss.item()])
                    plt.subplot(211)
                    plt.cla()
                    plt.plot([x[0] for x in perform_batch], [x[1]
                                                             for x in perform_batch], '.-')
                    plt.xlabel('batch_number')
                    plt.ylabel('average loss in current epoch')
                    plt.title('procession: %s of %s batches_total; %s epoch of %s epochs total' % (
                        k, len(train_iter), epoch, epoch_num))
                    plt.grid()
                    plt.pause(0.3)

            # epoch 结果打印
            self.learning_curve.append(
                [epoch, self.eval(train_iter), self.eval(val_iter)])
            plt.subplot(212)
            plt.cla()
            plt.plot([x[0] for x in self.learning_curve],
                     [x[1] for x in self.learning_curve],
                     '.-',
                     label='train')
            plt.plot([x[0] for x in self.learning_curve],
                     [x[2] for x in self.learning_curve],
                     '.-',
                     label='validation')
            plt.legend()
            plt.xlabel('epoch_num')
            plt.ylabel('evaliation criteria')
            plt.title('epoch summary')
            plt.grid()
            plt.pause(0.3)

            # early stop
            if self.judge_early_stop(early_stop_rounds):
                print('reached early_stop_rounds; training stop')
                return

    def judge_early_stop(self, early_stop_rounds):
        if early_stop_rounds is not None and len(self.learning_curve) > early_stop_rounds:
            tmp1 = [x[2] for x in self.learning_curve[-early_stop_rounds-1:-1]]
            tmp2 = [x[2] for x in self.learning_curve[-early_stop_rounds:]]
            diff = np.array(tmp2)-np.array(tmp1)
            if len(self.learning_curve) > early_stop_rounds and sum(diff > 0) == 0:
                return True
            else:
                return False
        else:
            return False

    def eval(self, val_iter):
        _, y_hat = self.predict(val_iter)
        y = []
        for batch in val_iter:
            y = y + list(batch[2].numpy())
        y = np.array(y)
        re = self.evalfun(y_hat, y)
        return re

    def predict(self, test_iter):
        self.net.eval()  # 将模型设置成预测模式
        array_y = None
        array_id = None
        with torch.no_grad():
            for batch in test_iter:
                ID = batch[0]
                X = batch[1].to(self.device)
                y_hat = self.net(X)
                if array_y is not None:
                    array_y = np.row_stack((array_y, y_hat.cpu().numpy()))
                    array_id = array_id+ID
                else:
                    array_y = y_hat.cpu().numpy()
                    array_id = ID
        return array_id, array_y


########################################################################################################################
#                                               Attention 模块
########################################################################################################################
class netAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key_seq, value_seq, query_seq):
        '''
        purpose:
            给定key,value,query 序列, 计算attention结果
        input:
            key_seq: [batch_size,seq_len,fea_dim]
            value_seq: [batch_size,seq_len,fea_dim]
            query_seq: [batch_size,seq_len,fea_dim]
        output:
            attention: [batch_size,seq_len of query_seq, fea_dim of value_seq]
        '''
        # 计算attention分数[batch_size,lenOfKey,lenOfQuery]
        alpha_seq = torch.bmm(key_seq,
                              query_seq.permute([0, 2, 1]))
        alpha_seq = F.softmax(alpha_seq, dim=1)  # 归一化

        # value_sum [batch_size,lenOfQuery,fea_dim of value]
        attention = torch.bmm(alpha_seq.permute([0, 2, 1]),
                              value_seq)
        return attention


########################################################################################################################
#                                               CRF 模块
########################################################################################################################
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)


        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

