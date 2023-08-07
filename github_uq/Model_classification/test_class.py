#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import random
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import FuncFormatter
import time
import os

date = time.strftime('%Y%m%d',time.localtime(time.time()))

def mkdir(path):
 
    folder = os.path.exists(path)

    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)             #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder for this day  ---")
    else:
        print("---  There is this folder!  ---")

mkdir(date)    

# In[11]:
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
    fig, ax = plt.subplots(figsize = (8, 8),dpi = 300)
    ax.plot(fpr, tpr, '-', color = '#ED7D31', linewidth = 3)
    ax.plot([0, 1], [0, 1], '--', color = 'black', markersize = 8)
    ax.set_xlabel('False Positive Rate', fontsize = 'xx-large')
    ax.set_ylabel('True Positive Rate', fontsize = 'xx-large')
    ax.text(0.0, 0.90, 'AUC: %.4f \nMCC: %.4f ' %(float(np.array(auc)),float(np.array(mcc))),fontsize = 'x-large')
    fig.tight_layout()
    fig.savefig(figure_file)


# In[9]:


def calculate_mcc(TP,FP,FN,TN):
    numerator = (TP * TN) - (FP * FN) #马修斯相关系数公式分子部分
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) #马修斯相关系数公式分母部分
    if denominator != 0:
        mcc = numerator/denominator
    else:
        mcc = 0
    return mcc

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# In[57]:
def plot_scatter(data_uncertainty,model_uncertainty):##展示模型和数据不确定性
    plt.figure()
    color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    x=[]
    y=[]
    for i in range(data_uncertainty.shape[0]):
        x.append(model_uncertainty[i])
        y.append(data_uncertainty[i])
    plt.xlabel('Model uncertainty')
    plt.ylabel('Data uncertainty')
    plt.scatter(x,y,c=color[2], alpha= 0.5)
    plt.show()

def plot_show_uncertainty(data_un,model_un):
    """
    :param x: 输入数据
    :return num_bin: 条柱数目
    """
    data_un=pd.DataFrame(data_un)
    data_static = data_un.describe().T
    model_un = pd.DataFrame(model_un)
    model_static = model_un.describe().T
    # data_IQR = data_static["75%"] - data_static["25%"]
    # data_bin_width = (2 * data_IQR) / np.power(data_IQR.shape[0], 1 / 3)
    data_x_max, data_x_min = data_static["max"]*10, data_static["min"]*10
    data_x_min = int(data_x_min)
    data_x_max = int(data_x_max)+1
    model_x_max, model_x_min = model_static["max"],model_static["min"]
    # if model_x_max > data_static["max"]:
    #     x_max = model_x_max
    # else:
    #     x_max = data_static["max"]
    # if model_x_min < data_static["min"]:
    #     x_min = model_x_min
    # else:
    #     x_min = data_static["min"]
    # # data_num_bin = int(((data_x_max - data_x_min) / data_bin_width) + 1)
    #  # 箱子数目
    plt.figure(dpi = 300)
    # plt.xlim(x_min, x_max)
    plt.hist(data_un, bins=20,color= 'darkblue', linewidth=0.5, alpha=0.5, label="data uncertainty")##根据情况画把 如果隔得很远分开画
    plt.hist(model_un, bins=50, color= 'darkred',linewidth=0.5, alpha=0.7, label="model uncertainty")
    plt.ylabel('Sample number', fontsize=15)
    plt.xlabel('Uncertainty value', fontsize=15)
    plt.legend()
    plt.show()

def plot_boxing(receive_data,reject_data):
    plt.figure(figsize=(5,10))
    plt.title('Examples Of Samples', fontsize=15)

    plt.boxplot([receive_data,reject_data], vert=True, showmeans=True)##可能要改
    plt.ylabel('Total uncertainty',fontsize=15)
    ax = plt.subplot()
    ax.set_xticklabels(['Receive', 'Reject'], fontsize=15)
    # plt.legend()
    plt.show()

def plot_boxing_reject():
    plt.figure(figsize=(80, 60),dpi = 300)
    fig, ax1 = plt.subplots()
    plt.title('Receive Samples', fontsize=20)
    ###
    a=[]
    a.append(np.load('./%s/1.00_receive_data_uncertainty.npy'%date))
    for i in range(10):
        print(i)
        i=1-0.05*(i+1)
        print(i)
        
        a.append(np.load('./%s/%.2f_receive_data_uncertainty.npy'%(date,i)))

    ###
    ax1.boxplot([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]], vert=True, showmeans=True,
                widths = 0.5,
                flierprops=dict(marker='.',markerfacecolor='green', markersize=5,linestyle='none'))  ##可能要改
    plt.ylabel('Data uncertainty', fontsize=15)
    plt.xlabel('%s Receive rate'%fp, fontsize=15)
    ax1.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50'], fontsize=10)
    
    ax2 = ax1.twinx()
    ax2.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50'], fontsize=10)
    ax2.plot((1,2,3,4,5,6,7,8,9,10,11), acc_array,
             color='darkred')
    ax2.set_ylabel('Accuracy', fontsize=15)
    
#     ax3 = ax1.twinx()
#     ax3.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50'], fontsize=10)
#     ax3.plot((1,2,3,4,5,6,7,8,9,10,11), auc,
#              color='darkblue')
#     ax3.set_ylabel('AUC_ROC', fontsize=15)
    
    fig.tight_layout()
    # plt.legend()
    plt.show()
    
def function_judge(choose_pred_mean,choose_sign,label):
    for i in range(choose_pred_mean.shape[0]):
        if choose_pred_mean[i] >= 0.5:
            choose_pred_mean[i]=1
        else:
            choose_pred_mean[i]=0
    percent_tn, percent_fp, percent_fn, percent_tp = (metrics.confusion_matrix(choose_pred_mean.astype(int),
                                                                               label[choose_sign].astype(
                                                                                   int),
                                                                               labels=[0, 1]).ravel()) / len(choose_sign)
    return percent_tn, percent_fp, percent_fn, percent_tp

def to_percent(temp,position):
    return '%1.0f' % (100 * temp) + '%'

def choose_sign_for_plot(uncertainty,y_sample,label):
    choose_sign=np.zeros((4,4))
    print('max='+str(uncertainty.max()))
    print('min=' + str(uncertainty.min()))
    # sort=np.sort(uncertainty)
    # threshold = list[round(0.90 * data_pred_entropy.shape[0])]
    # min=sort[0]
    # max=sort[-1]
    choose_sign1 = []
    choose_sign2 = []
    choose_sign3 = []
    choose_sign4 = []
    choose_sign5 = []
    for i in range(uncertainty.shape[0]):
        if uncertainty[i]<0.1:
            choose_sign1.append(i)
        if uncertainty[i]<0.2 and uncertainty[i]>0.1 :
            choose_sign2.append(i)
        if uncertainty[i] < 0.3 and uncertainty[i] > 0.2:
            choose_sign3.append(i)
        if uncertainty[i] < 0.4 and uncertainty[i] > 0.3:
            choose_sign4.append(i)

    choose_sign[0,:]= function_judge(np.mean(y_sample[:, choose_sign1], axis=0),choose_sign1,label)
    choose_sign[1,:]= function_judge(np.mean(y_sample[:, choose_sign2], axis=0),choose_sign2,label)
    choose_sign[2,:] = function_judge(np.mean(y_sample[:, choose_sign3], axis=0),choose_sign3,label)
    choose_sign[3,:] =function_judge(np.mean(y_sample[:, choose_sign4], axis=0),choose_sign4,label)
    plt.figure(dpi = 300)
    fig,ax1 = plt.subplots()
    for i in range(4):
        t=i*0.1
        percent_tn, percent_fp, percent_fn, percent_tp =  choose_sign[i,:]
        # plt.xlim(x_min, x_max)
        ax1.bar(t, percent_fn, width=0.1, align='edge', color='#FA7F6F',label='FN',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t, percent_fp, width=0.1, bottom=percent_fn, align='edge',color='#FFBE7A', label='FP',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t, percent_tn, width=0.1, bottom=percent_fp + percent_fn, align='edge',color='#82B0D2', label='TN',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t, percent_tp, width=0.1, bottom=percent_fp + percent_tn+percent_fn, align='edge',color='#275C9D', label='TP',linewidth = 0.1,edgecolor = '#08070F')
        if i==0:
            plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylabel('Percentage', fontsize=15)
    plt.xlabel('Data Uncertainty', fontsize=15)
    ax2 = ax1.twinx()
    ax2.plot((0.05,0.15,0.25,0.35),(len(choose_sign1),len(choose_sign2),len(choose_sign3),len(choose_sign4)),color='darkred')
    ax2.set_ylabel('Sample Number', fontsize=15)
    fig.tight_layout()
    plt.show()    
    
def testset(features, labels, n_splits, model_file, figure_file, log_file,percent_number):
    log = open(log_file, 'a')   

    assert len(features) == len(labels)
    
    input_dim = len(features[0])
    test_features = features
    test_labels = labels

    test_out = np.zeros(len(labels))
    
#     test_auc = []
#     test_precision = []
#     test_recall = []
#     test_mcc = []
#     test_acc = []
#     test_specificity = []
#     test_f1score = []
#     test_bacc = []
#     model_entropy_fps = []
    name = fp
    
    fold = 1
    
    test_model = LeNet(droprate = 0.2,input_dim = input_dim)
#     path = "../models_top2"
#     a = os.path.exists('%s_%d.pkl' % (model_file, fold))
#     if a == True:
    test_model.load_state_dict(torch.load('./choose_models/1_model_class_%s.pkl'%fp))
#     else:
#         print('%s_%d_nan!'% (model_file, i+1))
            
    test_model.eval()

    with torch.no_grad():
        test_out_prob = test_model(torch.from_numpy(test_features).float().unsqueeze(1))
        for i in range(len(test_out_prob)):
            if test_out_prob[i] >0.5:
                test_out[i] =1
            else:
                test_out[i] = 0


        
#         test_auc.append(test_auc_fold)
#         print(test_auc_fold)

        ####11.11uq
    
        enable_dropout(test_model)
        y_sample = np.zeros((10,len(test_labels)))
        drop_test_out = np.zeros(len(test_labels))

        for i in range(10):
            drop_test_out_prob = test_model(torch.from_numpy(test_features).float().unsqueeze(1))
            y_sample[i,:] =drop_test_out_prob.squeeze(-1)                   #得到10次抽样

        y_sample_mean =np.mean(y_sample,axis = 0)

        total_pred_entropy = -(y_sample_mean * np.log(y_sample_mean))         
        data_pred_entropy = np.mean(-(y_sample * np.log(y_sample)), axis=0)   
        model_pred_entropy = total_pred_entropy - data_pred_entropy           

        model_entropy = np.average(model_pred_entropy)
        plot_scatter(data_pred_entropy[:100],model_pred_entropy[:100])
        
        choose_sign_for_plot(data_pred_entropy, y_sample, labels)
        
        list=np.sort(data_pred_entropy)
        threshold=list[round(percent_number*len(data_pred_entropy))-1]#.shape[0]
        choose_sign=[]
        reject_sign =[]
        for i in range(data_pred_entropy.shape[0]):
            if data_pred_entropy[i]<threshold:
                choose_sign.append(i)
            else:
                reject_sign.append(i)
        choose_pred_mean=np.mean(y_sample[:,choose_sign],axis=0)
        reject_data_uncertainty = data_pred_entropy [reject_sign]
        receive_data_uncertainty = data_pred_entropy [choose_sign]
        
        if percent_number == 1.0:
            TP, TN, FP, FN = get_TPTNFPFN(test_out, test_labels)

            test_acc = metrics.accuracy_score(test_labels, test_out)
            test_precision = TP / (TP + FP)
            test_recall = metrics.recall_score(test_labels, test_out)
            test_specificity = TN / (FP + TN)
            test_mcc = calculate_mcc(TP,FP,FN,TN)
            test_f1score = f1_score(test_labels,test_out)
            test_bacc = metrics.balanced_accuracy_score(test_labels, test_out)

            fpr, tpr, thresholds = metrics.roc_curve(test_labels,test_out_prob) 

            test_auc = auc(fpr, tpr)
        if percent_number != 1.0:
            TP, TN, FP, FN = get_TPTNFPFN(test_out[choose_sign], test_labels[choose_sign])

            test_acc = metrics.accuracy_score(test_labels[choose_sign], test_out[choose_sign])
            test_precision = TP / (TP + FP)
            test_recall = metrics.recall_score(test_labels[choose_sign], test_out[choose_sign])
            test_specificity = TN / (FP + TN)
            test_mcc = calculate_mcc(TP,FP,FN,TN)
            test_f1score = f1_score(test_labels[choose_sign],test_out[choose_sign])
            test_bacc = metrics.balanced_accuracy_score(test_labels[choose_sign], test_out[choose_sign])

            fpr, tpr, thresholds = metrics.roc_curve(test_labels[choose_sign],test_out_prob[choose_sign]) 

            test_auc = auc(fpr, tpr)
        np.save('./%s/%.2f_receive_data_uncertainty.npy'%(date,percent_number), receive_data_uncertainty)

#             plot_boxing(receive_data_uncertainty,reject_data_uncertainty) #接受拒绝图 (分类用)
#             plot_scatter(data_pred_entropy,model_pred_entropy)# 散点图 （分类、回归用）
#             plot_show_uncertainty(data_pred_entropy,model_pred_entropy)# 柱状图 （分类回归用
        plot_show_uncertainty(data_pred_entropy,model_pred_entropy)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels[choose_sign], torch.as_tensor(choose_pred_mean))      
        print('reject predict AUC is ' + str(metrics.auc(fpr, tpr)))

            ####
    
#     test_auc_f = [[np.mean(test_auc)],[np.std(test_auc)]]
#     test_mcc_f = [[np.mean(test_mcc)],[np.std(test_mcc)]]
#     test_acc_f = [[np.mean(test_acc)],[np.std(test_acc)]]
#     test_precision_f = [[np.mean(test_precision)],[np.std(test_precision)]]
#     test_recall_f = [[np.mean(test_recall)],[np.std(test_recall)]]
#     test_f1score_f = [[np.mean(test_f1score)],[np.std(test_f1score)]]
#     test_specificity_f = [[np.mean(test_specificity)],[np.std(test_specificity)]]
#     test_bacc_f = [[np.mean(test_bacc)],[np.std(test_bacc)]]
#     test_uq_f = [[np.mean(model_entropy_fps)],[np.std(model_entropy_fps)]]
    
#     auc_max_index = test_auc.index(np.max(test_auc))
#     print('auc_max_index',auc_max_index)
    
#     print(test_mcc_f)
    # print(test_predicts_prob,test_labels_class)

#     metric = [test_auc_f,test_mcc_f,test_acc_f,test_precision_f,test_recall_f,test_specificity_f,test_f1score_f,test_bacc_f,test_uq_f,percent_number]
    metric = [name,test_auc, test_mcc, test_acc, test_precision, test_recall, test_specificity, test_f1score, test_bacc, model_entropy, percent_number]
    
    
    figure_test = True
    if figure_test:
        ROC_curve(fpr, tpr, test_auc,test_mcc, '%s.png' % (figure_file))

    print('test acc: %.4f, auc: %.4f, mcc: %.4f, precision: %.4f, recall: %.4f, specificity: %.4f, F1score: %.4f,test_bacc:%.4f,model_entropy:%.4f\n'
        % (test_acc, test_auc, test_mcc, test_precision,test_recall,test_specificity,test_f1score,test_bacc,model_entropy
))
    log.write('test acc: %.4f, auc: %.4f, mcc: %.4f, precision: %.4f, recall: %.4f, specificity: %.4f, F1score: %.4f,test_bacc:%.4f,model_entropy:%.4f\n'
        % (test_acc, test_auc, test_mcc, test_precision,test_recall,test_specificity,test_f1score,test_bacc,model_entropy
))
    log.close()

    return metric


# In[58]:


def main():
    global log_file
    log_file = './%s/1_class_%s.log' % (date,fp)
    
    log = open(log_file, 'w')
    print('Experimental data used is: "%s", the feature is: "%s"' % (i, fp))

    log.write('Experimental data used is: "%s", the feature is: "%s"\n' % (i, fp))
    log.close()
    
    model_file = 'choose_models/1_model_class_%s' % (fp)
    figure_file = '%s/1_class_test_%s' % (date,fp)
        
    metric = testset(test_features, test_labels, n_splits, model_file, figure_file, log_file,percent_number)
    
    
    return metric


        


# In[59]:


if __name__ == '__main__':
    test_labels = np.load('../../0_Data/0_class_data/data_split/3a4_test_labels.npy')#,allow_pickle = True)
    n_splits = 5
    
#     fp_name = ['morgan','Pubchem','AD2D','fcfp4','avalon','ecfp6','rdkit','APC','EStateFP','ExtFP','FP','GraphFP','MACCS','SubFP','SubFPC']
    fp_name = ['topconcate01','glconcate01','prconcate01']
    
    results = []
    for i in range(len(fp_name)):
        fp = fp_name[i]
#         n = model_number[i]
        
        test_features = np.load("../../0_Data/0_class_data/data_split/3a4_test_%s.npy"%fp,allow_pickle = True)
        test_features = test_features.astype(float)
        reject_pic = []
        for m in range(11):
            percent_number=1-(m)/20
            metric = main()
            
            results.append(metric)
            reject_pic.append(metric)
            
        reject_pic = pd.DataFrame(reject_pic,columns = ['fp','AUC-ROC','MCC','Accuracy','Precision','Recall',
                                                   'Specificity','F1-score','Balanced-ACC','Model Entropy','reject rate'])
        acc_array = np.array(reject_pic['Accuracy'])
        auc_array = np.array(reject_pic['AUC-ROC'])
        plot_boxing_reject() 

#         percent_number=1 
#         metric = main()
#         results.append(metric)
    results = pd.DataFrame(results,columns = ['fp','AUC-ROC','MCC','Accuracy','Precision','Recall','Specificity','F1-score','Balanced-ACC',
                                              'Model Entropy','reject rate'])
        
     
    results.to_csv('./%s/class_test_concat.csv'%date) 
        




