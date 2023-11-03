#!/usr/bin/env/pytorch python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable

from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score，accuracy_score，classification_report，balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os

# import argparse


# In[2]:

# class Residual(nn.Module):
#     def __init__(self,input_features,output_features):
#         super().__init__()
        
#         self.dense0 = nn.Linear(in_features = input_features,out_features = output_features)
# #         self.batchnorm0 = nn.BatchNorm1d(num_features = output_features)##
#         self.dropout0 = nn.Dropout(p = 0.4)
        
#         self.dense1 = nn.Linear(in_features = output_features,out_features = output_features)
# #         self.batchnorm1 = nn.BatchNorm1d(num_features = output_features)##
#         self.dropout1 = nn.Dropout(p = 0.4)
        
#         self.dense_skip = nn.Linear(in_features = input_features,out_features = output_features)
#     def forward(self,inputs):
#         l1 = nn.LeakyReLU()(inputs)
#         l1 = self.dense0(l1)
# #         l1 = self.batchnorm0(l1)
#         l1 = self.dropout0(l1)

#         l2 = nn.LeakyReLU()(l1)
#         l2 = self.dense1(l2)
# #         l2 = self.batchnorm1(l2)
#         l2 = self.dropout1(l2)
#         skip = self.dense_skip(inputs)
        
#         output = l2 + skip
#         return output

# class SimpleModelClassification(nn.Module):
#     def __init__(self, input_dim):
#         super(SimpleModelClassification, self).__init__()
#         self.hid1 = 64
#         self.hid2 = 16
        
#         self.FC1 = nn.Linear(input_dim,input_dim)
# #         self.BN0 = nn.BatchNorm1d(num_features = input_dim)##
#         self.FC2 = Residual(input_dim, input_dim)
#         self.FC3 = Residual(input_dim, self.hid1)
#         self.FC4 = Residual(self.hid1, self.hid2)
#         self.output = nn.Linear(self.hid2, 1) 
        
        
#     def forward(self, x):
#         x = self.FC1(x)
# #         x = self.BN0(x)
#         x = self.FC2(x)
#         x = self.FC3(x)
#         x = self.FC4(x)
        
#         x = self.output(x)
#         y = nn.Sigmoid()(x)
        
#         return y

# In[7]:
############################## 4.3
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class LeNet(nn.Module):
    def __init__(self, droprate=0.2,test=0,input_dim = 0):
        super(LeNet, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv1d(1, 20, kernel_size=5, padding=2))#(128,1, 1536)-(128,20, 1536)#1536是维度，需要改
        self.model.add_module('dropout1', nn.Dropout(p=droprate))
        self.model.add_module('maxpool1', nn.MaxPool1d(2, stride=2))#(128,1536,20)-(128,768,20) 
        self.model.add_module('conv2', nn.Conv1d(20, 50, kernel_size=5, padding=2))#(128,768,20)-(128,768,50)
        self.model.add_module('dropout2', nn.Dropout(p=droprate))
        self.model.add_module('maxpool2', nn.MaxPool1d(2, stride=2))#(128,768,50)-(128,384,50)
        self.model.add_module('flatten', Flatten()) #(128,384,50)-(128,384*50)
        self.model.add_module('dense3', nn.Linear(int(input_dim/4)*50 , 500))##################维度/4的整数
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('dropout3', nn.Dropout(p=droprate))
        self.final2= torch.nn.Linear(500, 3)
        self.final1= torch.nn.Linear(500, 1)

    def forward(self, x):
        out_value = self.model(x)  # linear output   ###############去哪一维度
        out = nn.Sigmoid()(self.final1(out_value))
        return out

############################## 4.3



def get_TPTNFPFN(y_pred, y_true):
    TP = 0; TN =0; FP = 0; FN = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_true[i] == 1:
                FN += 1
            else:
                TN += 1
        
    return TP, TN, FP, FN


# In[8]:


def ROC_curve(fpr, tpr, auc, mcc, figure_file):
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(fpr, tpr, '-', color = '#ED7D31')#, linewidth = 3)
    ax.plot([0, 1], [0, 1], '--', color = 'black', markersize = 8)
    ax.set_xlabel('False Positive Rate', fontsize = 'xx-large')
    ax.set_ylabel('True Positive Rate', fontsize = 'xx-large')
    ax.text(0.0, 0.90, 'AUC: %.2f\nMCC:%.2f' %(auc,mcc),fontsize = 'x-large')
    fig.tight_layout()
    fig.savefig(figure_file, dpi=300)


# In[9]:


def calculate_mcc(TP,FP,FN,TN):
    numerator = (TP * TN) - (FP * FN) #马修斯相关系数公式分子部分
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) #马修斯相关系数公式分母部分
    if denominator != 0:
        mcc = numerator/denominator
    else:
        mcc = 0
    return mcc


# In[10]:


def train(features, labels, n_splits, batch_size, learning_rate, epoch_total, epoch_log, optim_method, model_file, log_file):
    log = open(log_file, 'a')  
    global input_dim
    input_dim = len(features[0])

    best_train_acc_all = [] 
    best_train_auc_all = [] 
    best_train_mcc_all = []
    best_train_loss_all = []
    
    
    best_valid_acc_all = []
    best_valid_auc_all = []
    best_valid_mcc_all = []
    best_valid_loss_all = []
    
    

    kf = KFold(n_splits = n_splits, shuffle=True, random_state = 0)
    fold = 0
    for train_indices, valid_indices in kf.split(labels):    #输出索引
        fold += 1
        print('Fold %d%s' % (fold, '-' * 100))        
        log.write('Fold %d%s\n' % (fold, '-' * 100))
        
        train_features, train_labels = features[train_indices], labels[train_indices] 
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        

        ##########要改的报错，str不行，改成float
        train_set = TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long())
        train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)

        train_model = LeNet(droprate = 0.2,input_dim = input_dim)

        criterion = nn.BCELoss()

        if optim_method == 'Adam_set_weightdecay':
            optimizer = optim.Adam(train_model.parameters(), lr=learning_rate,weight_decay = 0.008)
        elif optim_method == 'Adam':
            optimizer = optim.Adam(train_model.parameters(), lr=learning_rate)
        elif optim_method == 'AdamW':
            optimizer = optim.AdamW(train_model.parameters(), lr=learning_rate)
        
        # best for all fold
        if fold == 1:
            best_train_acc = -1               
            best_valid_acc = -1
            best_valid_auc = -1
            best_valid_mcc = -1
                    
        # best for each fold
        best_train_acc_fold = -1
        best_train_auc_fold = -1
        best_train_mcc_fold = -1
        best_train_loss_fold = 100
        
        best_valid_acc_fold = -1
        best_valid_auc_fold = -1
        best_valid_mcc_fold = -1
        best_valid_loss_fold = 100
        
        #save loss/valuate value 
        train_loss_fold_batch = []
        train_loss_fold = []
        train_auc_fold = []
        train_acc_fold = []
        train_mcc_fold = []
        
        valid_loss_fold = []
        valid_auc_fold = []
        valid_acc_fold = []
        valid_mcc_fold = []
        
        valid_auc_top = 0.90

        for ep in range(epoch_total):
            for i, batch_data in enumerate(train_loader):
                batch_features, batch_labels = batch_data
                train_model.train()  
                
                batch_out = train_model(batch_features.unsqueeze(1))
                loss = criterion(batch_out.squeeze(), batch_labels.to(torch.float32))
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss_value = loss.data.detach().numpy()
                train_loss_fold_batch.append(train_loss_value)                
            if (ep+1) % epoch_log == 0:
                train_model.eval()
                with torch.no_grad():
                    #for train set
                    train_out = train_model(torch.from_numpy(train_features).float().unsqueeze(1))
                 
                    train_loss = criterion(train_out.squeeze(), torch.from_numpy(train_labels).to(torch.float32))
                    
                    train_predicts = np.where(np.array(train_out) > 0.5, 1, 0)
                    TP, TN, FP, FN = get_TPTNFPFN(train_predicts, train_labels) 
                    if (TP+FP) != 0:
                        # save train loss to list
                        train_loss_np = train_loss.data.detach().numpy()
                        train_loss_fold.append(train_loss_np)
#                         print(train_loss_fold)
                        
                        #save train acc auc
                        train_acc = metrics.accuracy_score(train_labels, train_predicts)
                        train_auc = metrics.roc_auc_score(train_labels, train_out)
                        train_acc_fold.append(train_acc)
                        train_auc_fold.append(train_auc)
                        
                        #skip the 3 valuation
#                         train_precision = TP / (TP + FP)
#                         train_recall = TP / (TP + TN)
                        train_mcc = calculate_mcc(TP,FP,FN,TN)
                        train_mcc_fold.append(train_mcc)
                    else:
                        continue
                  

                    # for valid set
                    valid_out = train_model(torch.from_numpy(valid_features).float().unsqueeze(1))
                    valid_loss = criterion(valid_out.squeeze(), torch.from_numpy(valid_labels).to(torch.float32))
                    
                    valid_predicts = np.where(np.array(valid_out) > 0.5, 1, 0)
                    
                    TP, TN, FP, FN = get_TPTNFPFN(valid_predicts, valid_labels)
                    if (TP+FP) != 0:
                        #save train loss to list
                        valid_loss_np = valid_loss.data.detach().numpy()
                        valid_loss_fold.append(valid_loss_np)
#                         print(valid_loss_fold)
                        valid_precision = TP / (TP + FP)
                        valid_recall = metrics.recall_score(valid_labels, valid_predicts)
                        valid_mcc = calculate_mcc(TP,FP,FN,TN)
                        #save valid acc auc    
                        valid_acc = metrics.accuracy_score(valid_labels, valid_predicts)
                        valid_auc = metrics.roc_auc_score(valid_labels, valid_out)
                        
                        valid_acc_fold.append(valid_acc)
                        valid_auc_fold.append(valid_auc)
                        valid_mcc_fold.append(valid_mcc)
                    else:
                        continue
                    
                    
                    

                    #更新准确率和AUC
#                     if valid_auc >= valid_auc_top:
#                         torch.save(train_model.state_dict(), '%s_%d_AUC%.4f_bc%s_lr%s.pkl' % (model_file,fold,valid_auc,bs,lr))
#                         valid_auc_top = valid_auc
                    
                    if valid_acc > best_valid_acc and valid_auc > best_valid_auc and valid_mcc > best_valid_mcc: 
                        best_valid_acc = valid_acc
                        best_valid_auc = valid_auc
                        best_valid_mcc = valid_mcc
                        best_valid_predicts = valid_predicts
                        

#                     if valid_acc > best_valid_acc_fold and valid_auc > best_valid_auc_fold and valid_mcc > best_valid_mcc_fold:
                    if valid_auc > best_valid_auc_fold and best_valid_loss_fold > valid_loss_np:
                        best_valid_acc_fold = valid_acc
                        best_valid_auc_fold = valid_auc
                        best_valid_mcc_fold = valid_mcc
                        best_valid_loss_fold = valid_loss_np
                        best_train_acc_fold = train_acc
                        best_train_auc_fold = train_auc
                        best_train_mcc_fold = train_mcc
                        best_train_loss_fold = train_loss_np
                        torch.save(train_model.state_dict(), '%s_%d_lossmin_bc%s_lr%s.pkl' % (model_file, fold,bs,lr))
                         
                    
                    print('  epoch: %04d, train loss:%.4f, valid loss: %.4f, valid acc: %.4f, auc: %.4f,mcc: %.4f, precision: %.4f, recall: %.4f'
                        % (ep+1, train_loss_np,valid_loss_np, valid_acc, valid_auc,valid_mcc, valid_precision, valid_recall))
                    log.write('  epoch: %04d, train loss:%.4f, valid loss: %.4f, valid acc: %.4f, auc: %.4f, mcc: %.4f,precision: %.4f, recall: %.4f\n'
                        % (ep+1, train_loss_np,valid_loss_np, valid_acc, valid_auc,valid_mcc, valid_precision, valid_recall))

        best_valid_acc_all.append(best_valid_acc_fold)
        best_valid_auc_all.append(best_valid_auc_fold)
        best_valid_mcc_all.append(best_valid_mcc_fold)
        best_valid_loss_all.append(best_valid_loss_fold)
        
        best_train_acc_all.append(best_train_acc_fold)
        best_train_auc_all.append(best_train_auc_fold)
        best_train_mcc_all.append(best_train_mcc_fold)
        best_train_loss_all.append(best_train_loss_fold)

        #results metrics of all 5 folds 
#         valid_acc_f = [np.mean(best_valid_acc_all),np.std(best_valid_acc_all)]
#         valid_auc_f = [np.mean(best_valid_auc_all),np.std(best_valid_auc_all)]
#         valid_mcc_f = [np.mean(best_valid_mcc_all),np.std(best_valid_mcc_all)]
#         valid_loss_f = [np.mean(best_valid_loss_all),np.std(best_valid_loss_all)]
#         train_acc_f = [np.mean(best_train_acc_all),np.std(best_train_acc_all)]
#         train_auc_f = [np.mean(best_train_auc_all),np.std(best_train_auc_all)]
#         train_mcc_f = [np.mean(best_train_mcc_all),np.std(best_train_mcc_all)]
#         train_loss_f = [np.mean(best_train_loss_all),np.std(best_train_loss_all)]
        
        
        valid_acc_max = [np.max(best_valid_acc_all),np.argmax(best_valid_acc_all)]
        valid_auc_max = [np.max(best_valid_auc_all),np.argmax(best_valid_acc_all)]
        valid_mcc_max = [np.max(best_valid_mcc_all),np.argmax(best_valid_acc_all)]
        valid_loss_max = [np.max(best_valid_loss_all),np.argmax(best_valid_acc_all)]
        train_acc_max = [np.max(best_train_acc_all),np.argmax(best_valid_acc_all)]
        train_auc_max = [np.max(best_train_auc_all),np.argmax(best_valid_acc_all)]
        train_mcc_max = [np.max(best_train_mcc_all),np.argmax(best_valid_acc_all)]
        train_loss_max = [np.max(best_train_loss_all),np.argmax(best_valid_acc_all)]
        
        metric = [train_acc_max, train_auc_max, train_mcc_max, train_loss_max,valid_acc_max, valid_auc_max, valid_mcc_max, valid_loss_max]
       #一套参数下(model).一折内（fold），所有epoch连在一起，输出一个npy        
#         train_loss_batch = np.array(train_loss_fold_batch)    
#         train_loss_fold = np.array(train_loss_fold)
#         train_auc_fold = np.array(train_auc_fold)
#         train_acc_fold = np.array(train_acc_fold)
#         train_mcc_fold = np.array(train_mcc_fold)
#         valid_loss_fold = np.array(valid_loss_fold)
#         valid_acc_fold = np.array(valid_acc_fold)
#         valid_auc_fold = np.array(valid_auc_fold)
#         valid_mcc_fold = np.array(valid_mcc_fold)

#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_train_loss_iter.npy'%(fp,n,fold),train_loss_batch) 
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_train_loss_epoch.npy'%(fp,n,fold),train_loss_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_train_auc_epoch.npy'%(fp,n,fold),train_auc_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_train_acc_epoch.npy'%(fp,n,fold),train_acc_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_train_mcc_epoch.npy'%(fp,n,fold),train_mcc_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_valid_loss_epoch.npy'%(fp,n,fold),valid_loss_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_valid_auc_epoch.npy'%(fp,n,fold),valid_auc_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_valid_acc_epoch.npy'%(fp,n,fold),valid_acc_fold)
#         np.save('./arg_optim/3a4_class_%s_model%s_fold%s_valid_mcc_epoch.npy'%(fp,n,fold),valid_mcc_fold)

    print('-' * 99)
    print('  best_valid_acc: %.4f, best_valid_auc: %.4f,best_valid_mcc: %.4f' % (best_valid_acc, best_valid_auc, best_valid_mcc))
    
    print('  mean and std of valid acc for %d-fold cross validation are: %.4f, %.4f'
         % (n_splits, np.mean(best_valid_acc_all), np.std(best_valid_acc_all)))
    print('  mean and std of valid auc for %d-fold cross validation are: %.4f, %.4f'
         % (n_splits, np.mean(best_valid_auc_all), np.std(best_valid_auc_all)))
    print('  mean and std of valid mcc for %d-fold cross validation are: %.4f, %.4f'
         % (n_splits, np.mean(best_valid_mcc_all), np.std(best_valid_mcc_all)))

    log.write('%s\n' % ('-' * 100))
    log.write('  best_valid_acc: %.4f, best_valid_auc: %.4f, best_valid_mcc: %.4f\n' % (best_valid_acc, best_valid_auc, best_valid_mcc))
    log.write('  mean and std of valid auc for %d-fold cross validation are: %.4f, %.4f\n'
              % (n_splits, np.mean(best_valid_auc_all), np.std(best_valid_auc_all)))
    log.write('  mean and std of valid acc for %d-fold cross validation are: %.4f, %.4f\n'
              % (n_splits, np.mean(best_valid_acc_all), np.std(best_valid_acc_all)))
    log.write('  mean and std of valid mcc for %d-fold cross validation are: %.4f, %.4f\n'
              % (n_splits, np.mean(best_valid_mcc_all), np.std(best_valid_mcc_all)))
    log.close()
    
    return metric

# In[14]:


def main():
    global log_file
    log_file = 'logs/1_class_%s_%s.log' % (fp,n)
    
    log = open(log_file, 'w')
    print('Input information are')
    print('Experimental data used is: "%s", the feature is: "%s"' % ('train set', fp))
    print('Batch size: "%s", learning_rate: "%s", number of epoch: "%s"' % (batch_size, learning_rate, epoch_total))
    print('Optimize method: "%s"' % optim_method)

    log.write('Input information are\n')
    log.write('Experimental data used is: "%s", the feature is: "%s"\n' % ('train set', fp))
    log.write('Batch size: "%s", learning_rate: "%s", number of epoch: "%s"\n'% (batch_size, learning_rate, epoch_total))
    log.write('Optimize method: "%s"\n' % optim_method)
    log.close()
    
    model_file = 'models/1_model_class_%s_%s' % ( fp,n)
    figure_file = 'figures/1_class_%s_%s' % (fp,n)
        
    metric = train(train_features, train_labels, n_splits, batch_size, learning_rate, epoch_total, epoch_log, optim_method, model_file, log_file)
    
    
    return metric


# In[15]:


if __name__ == '__main__':
    train_labels = np.load('../0_Data/0_class_data/data_split/3a4_train_labels.npy')#,allow_pickle = True)
    train_labels = train_labels.astype(np.float32)
    task_num = 2
    fp_name = ['topconcate01','glconcate01']
    results = {}

    for fp in fp_name:
        features = np.load("../0_Data/0_class_data/data_split/3a4_train_%s.npy"%fp)
        train_features = features.astype(np.float32)
        n = 0

        batch_size_choice = [256]#16,32,64,128,256]#64,128
        learning_rate_choice = [0.001]#,0.0001] #0.001,0.0001]
        epoch_total_choice = [100]   #100,200,300]
        epoch_log_choice = [1]
        optim_method_choice = ['Adam']  #'Adam_set_weightdecay',,
                
        for bs in batch_size_choice:
            batch_size = bs
            for lr in learning_rate_choice:
                learning_rate = lr
                for et in epoch_total_choice:
                    epoch_total = et
                    epoch_log = 1
                    for op in optim_method_choice:
                        n_splits = 5  #################################################
                        optim_method = op
                        n = n+1
                        metric = main()
                        results['%s,%s,%s'%(fp,bs,lr)] = [x[0]for x in metric]
                        results['%s,%s,%s,fold'%(fp,bs,lr)] = [x[1]for x in metric]
                        
                        
        #将每一类分子表征的最好调参结果存成表格，所有的结果汇到一张表格里，并画成柱状图
        #为保证选取的auc和mcc是一组的，将数组加起来判断和最大的为最优的（有待改进）——弃用
        #param_test_evaluate = param_test_auc + param_test_mcc
       
       
        #存成表格
    DataFrame = pd.DataFrame(results,index = ['train_acc_max', 'train_auc_max', 'train_mcc_max', 'train_loss_max','valid_acc_max', 'valid_auc_max', 'valid_mcc_max', 'valid_loss_max'])
    DataFrame.T.to_csv("./results/3a4_class_concat_results_%s.csv"%task_num)
    