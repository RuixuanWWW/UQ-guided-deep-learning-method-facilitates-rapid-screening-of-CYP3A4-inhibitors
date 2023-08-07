import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import seaborn as sns
import math
import torch
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
from deep_evidential_regression_loss_pytorch import EvidentialLossNLL

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,TensorDataset
from matplotlib.ticker import FuncFormatter

import time
import os

date = time.strftime('%Y%m%d_2_retrain_02',time.localtime(time.time()))
# date = '20230614_2'

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def mkdir(path):
 
    folder = os.path.exists(path)

    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)             #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder for this day  ---")
    else:
        print("---  There is this folder!  ---")

mkdir(date)   
mkdir('./%s/backups/'%date)
mkdir('./%s/models/'%date)
mkdir('./%s/test/'%date)
mkdir('./%s/pics/'%date)
mkdir('./%s/calibration/'%date)



device = ('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self, droprate=0.2,test=0):
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
        self.model.add_module('linear',nn.Linear(500,10))
        self.final2= torch.nn.Linear(11, 3)
        self.final1= torch.nn.Linear(11, 1)
        # self.model.add_module('final2', nn.Linear(50, 4))
        # self.model.add_module('sigmoid', nn.Sigmoid())
        # self.model.add_module('final2',nn.ReLU(inplace=False))
        ###11.12
        # self.final = torch.nn.Linear(50, 4)
        # self.get_var = torch.nn.Linear(500, 1)

    def forward(self, x):

        # x1=self.model(x)
        # x2=self.model(x)
        # print('x1='+str(x1))
        # print('x2=' + str(x2))

        ###11.12
        # y=self.model(x)
        # mu = self.final(y)
        # sigma = self.get_var(y)
        out_value = self.model(x)  # linear output   ###############去哪一维度
        out_value_c = torch.cat([out_value,x[:,:,-1]],axis = 1)
#         print(x[:,:,-1].size())
        out_probability = torch.abs(self.final2(out_value_c))
        out = torch.cat((self.final1(out_value_c), out_probability), dim=1)
        return out


class LeNetRegressor:
    def __init__(self, droprate=0.0, activation='relu', \
                 max_epoch=1000000, lr=0.001, weight_decay=1e-6,bc = 128,input_dim = 0,lam = 0.2,epsilon = 1e-4):
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = bc
        self.model = LeNet(droprate=droprate)
        self.criterion = EvidentialLossNLL()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.to(device)
        self.criterion.to(device)

    def fit(self, X_train, y_train, lam,epsilon,verbose=True):
        valid_result = np.zeros(8)
        
        X, X_valid, y, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle = False)
        
#         X = Variable(torch.from_numpy(X_train_).type(torch.FloatTensor)).to(device)
#         y = Variable(torch.from_numpy(y_train_).type(torch.FloatTensor)).to(device)
        
#         X_valid = Variable(torch.from_numpy(X_valid).type(torch.FloatTensor)).to(device)
#         y_valid = Variable(torch.from_numpy(y_valid).type(torch.FloatTensor)).to(device)
        
        train_set = TensorDataset(X, y) 
        train_loader = DataLoader(dataset = train_set,batch_size = self.batch_size,shuffle = True)
        
        best_loss = 5
        best_rmse = 1
        
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        # X_test, y_test = iter(testloader).next()
        # X_test = X_test.to(device)
        print(self.model)
        for epoch in range(self.max_epoch):
            for i,batch_loader in enumerate(train_loader):
                batch_features,batch_values = batch_loader
                self.model.train()
                self.optimizer.zero_grad()
    #             print(X.unsqueeze(1).shape)
                outputs = self.model(batch_features.unsqueeze(1))
    #             print(outputs)
                loss = self.criterion(outputs, batch_values,lam,epsilon)
                loss.backward()
                self.optimizer.step()
                loss_train_value = loss.data.detach().cpu().numpy()
    #             print(loss.shape)
    #             print(loss.cpu().data)

    #             print(y.size(),outputs.size())
    #             print(outputs.data)

                rmse = get_Rmse(outputs.cpu().detach().numpy()[:,0],y.cpu().numpy())
                r2 = r2_score(batch_values.cpu(), outputs.cpu().detach().numpy()[:,0])
                r,p = pearsonr(batch_values.cpu(),outputs.cpu().detach().numpy()[:,0])
                
                if verbose:
                    print('Epoch {} loss: {}'.format(epoch + 1, self.loss_[-1]))
                # y_test_pred = self.predict(X_test.unsqueeze(1)).cpu()
            if (epoch+1)%1 == 0:
                self.model.eval()
                with torch.no_grad():
                    outputs_valid = self.model(X_valid.unsqueeze(1))
                    loss_valid = self.criterion(outputs_valid, y_valid,lam,epsilon)
                    loss_valid_value = loss_valid.data.detach().cpu().numpy()
                    rmse_valid = get_Rmse(outputs_valid.cpu().detach().numpy()[:,0],y_valid.cpu().numpy())
                    print(' epoch: %s, train loss: %.4f, valid loss: %.4f,train RMSE: %.4f,valid RMSE: %.4f'
                       % (epoch+1,loss_train_value, loss_valid_value,rmse, rmse_valid))
                    
                    if rmse_valid<best_rmse and loss_valid_value<best_loss:
                        best_rmse = rmse_valid
                        best_loss = loss_valid_value
                        torch.save(lenet.model,'./%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate))
                        best_r_valid,p = pearsonr(y_valid.cpu().numpy(), outputs_valid.cpu().detach().numpy()[:,0])
                        best_r2_valid = r2_score(y_valid.cpu().numpy(),outputs_valid.cpu().detach().numpy()[:,0])
                        valid_result[0] = rmse
                        valid_result[1] = r
                        valid_result[2] = r2
                        valid_result[3] = loss_train_value
                        valid_result[4] = best_rmse
                        valid_result[5] = best_r_valid
                        valid_result[6] = best_r2_valid
                        valid_result[7] = best_loss
        valid_results['fp:%s,lam:%s,bc: %s,lr: %s'%(fp,lam,batchsize,learning_rate)] = list(valid_result)                    
        return self





class build_dataset(): #需要继承Dataset类
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx,:]
        return data, label

def enable_dropout(model):####根据代码新加入 在predict 环节使用了新加的dropout层
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_Rmse(data,label):
    mse=0
    for i in range(data.shape[0]):
        mse+=(data[i]-label[i])**2
    return math.sqrt(mse/data.shape[0])

def calculate_variance(data,mean):

    total=[]
    for k in range(data.shape[1]):
        variance = 0
        for i in range(data.shape[0]):
            variance += (data[i,k] - mean[k])**2
        total.append(variance/data.shape[0])
    return total

def get_MC_samples(network, X, dropout=True, mc_times=64):
    pred_v = [];
    a_u = []

    for t in range(mc_times):
        if dropout:
            prediction, var = network(X)
        else:
            network.eval()
            prediction, var = network(X)

        pred_v.append(prediction.data.numpy())
        a_u.append(var.data.numpy())

    pred_v = np.array(pred_v);
    a_u = np.array(a_u)
    a_u = np.sqrt(np.exp(np.mean(a_u, axis=0)))
    pred_mean = np.mean(pred_v, axis=0)
    e_u = np.sqrt(np.var(pred_v, axis=0))
    save_var = pd.DataFrame(a_u.squeeze())
    pd.DataFrame(save_var).to_csv('var_test_data.csv')
    return pred_mean.squeeze(), a_u.squeeze(), e_u.squeeze()

def plot_show_uncertainty(data_un,model_un):
    """
    :param x: 输入数据
    :return num_bin: 条柱数目
    """
    data_un=pd.DataFrame(data_un)
    data_static = data_un.describe().T
    print(data_static)
#     data_un = data_un[data_un<np.max(data_un)]
#     data_un = data_un[data_un<np.max(data_un)]
#     data_un = data_un[data_un<np.max(data_un)]
#     data_un = data_un[data_un<np.max(data_un)]
#     data_un = data_un[data_un<np.max(data_un)]
    
    
    model_un = pd.DataFrame(model_un)
    model_static = model_un.describe().T
    print(model_static)
    
#     model_static = model_un.describe().T
#     # data_IQR = data_static["75%"] - data_static["25%"]
#     # data_bin_width = (2 * data_IQR) / np.power(data_IQR.shape[0], 1 / 3)
#     data_x_max, data_x_min = data_static["max"]*10, data_static["min"]*10
#     data_x_min = int(data_x_min)
#     data_x_max = int(data_x_max)+1
#     model_x_max, model_x_min = model_static["max"],model_static["min"]
#     # if model_x_max > data_static["max"]:
    #     x_max = model_x_max
    # else:
    #     x_max = data_static["max"]
    # if model_x_min < data_static["min"]:
    #     x_min = model_x_min
    # else:
    #     x_min = data_static["min"]
    # # data_num_bin = int(((data_x_max - data_x_min) / data_bin_width) + 1)
    #  # 箱子数目
    plt.figure(figsize = (10, 9),dpi = 500)
    font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica', 
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 30,
        }
    plt.tick_params(labelsize=30)
    bwith = 3
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # plt.xlim(x_min, x_max)
    plt.hist(data_un, bins=50,color= '#275C9D', linewidth=0.5, alpha=0.8, label="Data UQ")##根据情况画把 如果隔得很远分开画
    plt.hist(model_un, bins=18, color= '#A42630',linewidth=0,alpha=0.7,label="Model UQ")#
    
    plt.ylabel('count of samples',size = 30)
#     plt.xlabel('value',size = 20)
    
    font_legend = {'family': 'sans-serif',
         'weight': 'normal',
         'size': 20}
    plt.legend(
    #     loc = 'lower right', # 图例位置
        frameon = True, # 去除图例边框
    #     facecolor = 'orange',
    #     edgecolor = 'blue', # 设置边框颜色，边框设置为白色则和无边框效果相似
        prop = font_legend)
    plt.show()
    
def plot_boxing(receive_data,reject_data):
    plt.figure(figsize=(5,10))
    plt.title('Examples Of Samples', fontsize=15)
    ###

    ###
    plt.boxplot([receive_data,reject_data], vert=True, showmeans=True)##可能要改
    plt.ylabel('Total uncertainty',fontsize=15)
    ax = plt.subplot()
    ax.set_xticklabels(['Receive', 'Reject'], fontsize=15)
    # plt.legend()
    plt.show()


def plot_boxing_reject(rmse_arr):
    plt.figure(figsize=(32, 24),dpi = 500)
    fig, ax1 = plt.subplots()
#     plt.title('Received Samples', fontsize=20)
    bwith = 1.5
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(0)
    
    ###
    a=[]
    full = np.load('./%s/backups/1.00_receive_data_uncertainty.npy'%date)
#     full_a = full[full<max(full)]
#     full_b = full_a[full_a<max(full_a)]

#     full_c = full_b[full_b<max(full_b)]
#     full_d = full_c[full_c<max(full_c)]
#     full_e = full_d[full_d<max(full_d)]
    
#     a.append(full_e)
    a.append(full)
    
    for i in range(10):
        i=1-0.05*(i+1)
        a.append(np.load('./%s/backups/%.2f_receive_data_uncertainty.npy'%(date,i)))

    ###
    print(len(a[0]),len(a))
    ax1.boxplot([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]], vert=True, showmeans=True,
                widths = 0.5,flierprops=dict(marker='.',markerfacecolor='darkgreen', markersize=5,linestyle='none'))  ##可能要改
    
    plt.ylabel('Data uncertainty', fontsize=20)
    plt.xlabel('Reception rate', fontsize=20)
#     ax1.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70'], fontsize=10)
    ax1.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50'], fontsize=10)
    
    ax2 = ax1.twinx()
#     ax2.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70'], fontsize=10)
    ax2.set_xticklabels(['1.00', '0.95', '0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50'], fontsize=10)
    
    ax2.plot((1,2,3,4,5,6,7,8,9,10,11), rmse_arr,
             color='darkred')
    ax2.set_ylabel('RMSE', fontsize=20)
    ax2.spines['right'].set_linewidth(2)
    
    ax2.spines['right'].set_color('darkred')
    
    fig.tight_layout()
    # plt.legend()
    plt.show()

def function_judge(choose_pred_mean,choose_sign,label):
#     print(np.max(np.absolute(choose_pred_mean-label)))
#     print(np.min(np.absolute(choose_pred_mean-label)))
    diff = np.absolute(choose_pred_mean-label[choose_sign])
    sns.distplot(diff,label = '%s'%len(diff))
    plt.legend()
    diff_df=pd.DataFrame(diff)
    diff_static = diff_df.describe().T
    print('diff_static')
    print(diff_static)
    
    diff_type = choose_pred_mean
    for i in range(choose_pred_mean.shape[0]):
        if diff[i]<=0.1:
            diff_type[i]=0
        if diff[i]<0.3 and diff[i]>=0.1:
            diff_type[i]=1
        if diff[i]<0.5 and diff[i]>=0.3:
            diff_type[i]=2
        if diff[i]>=0.5:
            diff_type[i]=3
#     print(np.sum(choose_pred_mean ==0))
#     print(len(choose_sign))
    print([np.sum(diff_type ==0),np.sum(diff_type ==1),np.sum(diff_type  ==2),np.sum(diff_type ==3)])
    percent_tn, percent_fp, percent_fn, percent_tp = np.array([np.sum(diff_type ==0),np.sum(diff_type ==1),np.sum(diff_type  ==2),np.sum(diff_type ==3)]) / len(choose_sign)
    return percent_tn, percent_fp, percent_fn, percent_tp

def to_percent(temp,position):
    return '%1.0f' % (100 * temp) + '%'

def choose_sign_for_plot(uncertainty,y,label):
    num = 0.2
  
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
        if uncertainty[i]<0.6:
            choose_sign1.append(i)
        if uncertainty[i]<0.8 and uncertainty[i]>=0.6 :
            choose_sign2.append(i)
        if uncertainty[i] <1.0 and uncertainty[i] >=0.8:
            choose_sign3.append(i)
        if uncertainty[i] >=1.0:
            choose_sign4.append(i)

    choose_sign[0,:]= function_judge(y[choose_sign1],choose_sign1,label)
    choose_sign[1,:]= function_judge(y[choose_sign2],choose_sign2,label)
    choose_sign[2,:] = function_judge(y[choose_sign3],choose_sign3,label)
    choose_sign[3,:] =function_judge(y[choose_sign4],choose_sign4,label)
    plt.figure(dpi = 500)
    fig,ax1 = plt.subplots(figsize = (8, 6))
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(0)
    
    for i in range(4):
        t=i*num
        percent_tn, percent_fp, percent_fn, percent_tp = choose_sign[i,:]
        print(percent_tn, percent_fp, percent_fn, percent_tp)
        # plt.xlim(x_min, x_max)
        
        ax1.bar(t+0.4, percent_tn, width=num, align='edge', color='#FA7F6F',label='0-0.1',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t+0.4, percent_fp, width=num, bottom=percent_tn, align='edge',color='#FFBE7A', label='0.1-0.3',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t+0.4, percent_fn, width=num, bottom=percent_fp + percent_tn, align='edge',color='#82B0D2', label='0.3-0.5',linewidth = 0.1,edgecolor = '#08070F')
        ax1.bar(t+0.4, percent_tp, width=num, bottom=percent_fp + percent_tn+percent_fn, align='edge',color='#275C9D', label='>=0.5',linewidth = 0.1,edgecolor = '#08070F')
        if i==0:
            plt.legend(loc = 'upper right',framealpha = 0.55,title = 'Absolute Error',title_fontsize = 15)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylabel('Percentage', fontsize=25)
    plt.xlabel('Data uncertainty', fontsize=25)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.ylim(0,1)
    plt.xlim(0.4,1.2)
    
    ax2 = ax1.twinx()
    ax2.plot((num/2+0.4,3*num/2+0.4,5*num/2+0.4,7*num/2+0.4),(len(choose_sign1),len(choose_sign2),len(choose_sign3),len(choose_sign4)),color='darkred')
    ax2.set_ylabel('Count of samples', fontsize=25)
    plt.yticks(size = 20)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['right'].set_color('darkred')
    
    fig.tight_layout()
    plt.show()

    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):#n=len(class_uq
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.4f},{b:.4f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap       
    
    
def predict_reg(model, x, label,lam,T=10):###原无label
    X = Variable(x.type(torch.FloatTensor).to(device))
    model.eval() ####
    
    output = model(X.unsqueeze(1)).data.cpu()
    label = label.cpu().numpy()

    min_val = 1e-6
    # Split the outputs into the four distribution parameters
    mu, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
    v = torch.nn.Softplus()(loglambdas) + min_val
    alpha = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
    beta = torch.nn.Softplus()(logbetas) + min_val
    
    mu= mu.squeeze().numpy()
    v = v.squeeze().numpy()
    alpha = alpha.squeeze().numpy()
    beta = beta.squeeze().numpy()
    
    data_uncertainty= beta/(alpha-1)
    model_uncertainty=beta/(alpha-1)/v
    total_uncertainty = data_uncertainty + model_uncertainty
    
    sigma = np.sqrt(data_uncertainty)
    inverse_evidense = 1/(alpha-1)/v
    
    print('sigma',len(sigma),sigma[:10])
    np.save('./%s/calibration/%s_param(lam%s_bc%s_lr%s)_%s_pred.npy'%(date,fp,lam,batchsize,learning_rate,number),mu)
    np.save('./%s/calibration/%s_param(lam%s_bc%s_lr%s)_%s_datauq.npy'%(date,fp,lam,batchsize,learning_rate,number),data_uncertainty)
    np.save('./%s/calibration/%s_param(lam%s_bc%s_lr%s)_%s_modeluq.npy'%(date,fp,lam,batchsize,learning_rate,number),model_uncertainty)
    
    np.save('./%s/calibration/%s_param(lam%s_bc%s_lr%s)_%s_sigma.npy'%(date,fp,lam,batchsize,learning_rate,number),sigma)
    np.save('./%s/calibration/%s_param(lam%s_bc%s_lr%s)_%s_ie.npy'%(date,fp,lam,batchsize,learning_rate,number),inverse_evidense)
    

    result = []
    test_result = []
    for m in range(3):
        if m==0:
            uncertainty= model_uncertainty
            type = str('model')
        elif m==1:
            uncertainty = data_uncertainty
            type = str('data')
        else:
            uncertainty = total_uncertainty
            type = str('total')
        list = np.sort(uncertainty, axis=0)
        choose_sign=[]
        
        threshold = list[round(number * uncertainty.shape[0])-1]
        for i in range(uncertainty.shape[0]):
            if uncertainty[i] <= threshold:
                choose_sign.append(i)
        # choose_pred_mean=np.mean(y_sample[:,choose_sign],axis=0)
        Rmse = get_Rmse(mu[choose_sign], label[choose_sign])
        reject_r, p = pearsonr(label[choose_sign], mu[choose_sign])
        reject_R_squre = r2_score(label[choose_sign], mu[choose_sign])
        
        result.append(Rmse)
        result.append(reject_r)
        result.append(reject_R_squre)

        if number == 1 and m ==0:
            test_result.append(Rmse)
            test_result.append(reject_r)
            test_result.append(reject_R_squre)
            test_result.append(np.mean(uncertainty))
#             plot_show_uncertainty(data_uncertainty,model_uncertainty)####

        if number == 1 and m ==1:
            test_result.append(np.mean(uncertainty))
#             choose_sign_for_plot(uncertainty, mu, label)

#             print('!!! para_result',para_result)
#             print(batchsize,learning_rate)
#         else:
#             print('error')
            test_results['fp:%s,lam:%s,bc: %s,lr: %s'%(fp,lam,batchsize,learning_rate)] = test_result
#             initial_file['regre. pred.(fp:%s,bc: %s,lr: %s)'%(fp,batchsize,learning_rate)] = mu
#             initial_file['regre. uq(fp:%s,bc: %s,lr: %s)'%(fp,batchsize,learning_rate)] = uncertainty
            make_colors = np.zeros(len(uncertainty))
            for i in range(len(uncertainty)):
                if uncertainty[i] >=0.9:
                    make_colors[i] = 0
                else:
                    make_colors[i] = 1
            
            
            figure_test = False
            if figure_test:
#                 cmap_ = plt.get_cmap("RdBu")
#                 tcmap = truncate_colormap(cmap_, 0.9,0.01) #错 对
        #         print(type(test_predicts),type(test_labels))
                Real_Labels = np.array(label)#.tolist()
                print(len(label))
                Pred_Labels = np.array(mu)#.tolist()
                make_a_picture = {"Exp. Labels":Real_Labels,"Pred. Labels":Pred_Labels,"colors":make_colors}
                sns.set(style="white",font_scale=1.5)
                g = sns.jointplot(x="Exp. Labels", y='Pred. Labels', data=make_a_picture,color='#196BA3',ratio=5,marginal_kws=dict(bins=15, kde=True,color='#196BA3'),kind = "reg")
                plt.xlabel("Real  Labels")
                plt.ylabel("Pred  Labels")
                g.ax_joint.text(3.2,8.0,'r: %.3f\nRMSE: %.3f'%(np.mean(reject_r),np.mean(Rmse)))#
                g.fig.set_size_inches(6,6)
                plt.savefig('./%s/points.png'% date, dpi=300)
        
        print('---------fp:%s---bc: %s---lr: %s---type:%s---------'%(fp,batchsize,learning_rate,type))
        print('%.2f , R=%.4f'%(number,reject_r))
        print('%.2f , RMSE=%.4f '%(number,Rmse))
        print('%.2f , R_squre=%.4f '%(number,reject_R_squre))
    
    results['%s'%number] = result


# fp_choice = ['avalon','ecfp6','fcfp4','morgan','rdkit','AD2D','APC','EStateFP','ExtFP','FP','GraphFP','MACCS','Pubchem','SubFP','SubFPC']#
fp_choice = ['top_100']#'topconcate01','glconcate01',prconcate01,'glconcate01','prconcate01'

test_results = {}
valid_results = {}
   
for fp in fp_choice:
    
    pre_train_data = np.load('../0_Data/0_regre_3a4/data_split/3a4_train_%s.npy'%fp)
    pre_test_data = np.load('../0_Data/0_regre_3a4/data_split/3a4_test_%s.npy'%fp)
    
    pre_train_label = np.load('../0_Data/0_regre_3a4/data_split/3a4_train_labels.npy')
    pre_test_label = np.load('../0_Data/0_regre_3a4/data_split/3a4_test_labels.npy')
    
    pre_train_data=pre_train_data.astype(np.float32)
    pre_test_data=pre_test_data.astype(np.float32)
    pre_train_label = pre_train_label.astype(np.float32)
    pre_test_label = pre_test_label.astype(np.float32)
    
    pre_train_data = Variable(torch.from_numpy(pre_train_data).type(torch.FloatTensor)).to(device)
    pre_train_label = Variable(torch.from_numpy(pre_train_label).type(torch.FloatTensor)).to(device)

    pre_test_data = Variable(torch.from_numpy(pre_test_data).type(torch.FloatTensor)).to(device)
    pre_test_label = Variable(torch.from_numpy(pre_test_label).type(torch.FloatTensor)).to(device) 
    
    
    train_dataset=build_dataset(pre_train_data,pre_train_label)
    test_dataset=build_dataset(pre_test_data,pre_test_label)
    
#     initial_file = pd.read_csv('../0_Data/0_regre_3a4/data_split/3a4_880_uq.csv')

    

    global number,maxepoch,input_dim#,batchsize,learning_rate
    input_dim = len(pre_train_data[0])
    number=1
    # for k in range(3):
    #     maxepoch=1000*(2**k)
    lambs = [0.1,0.2]#0.15,0.2,0.25,0.3]#[0.005,0.01,0.05,0.1,0.2,0.4,0.6,1,2]
    epsilon = 1e-4
    bc_choice = [16,32,64,128]#,32,64]#16,,51216,,64,128,256,64,128
    lr_choice = [0.01,0.001]#,0.01,0.0001,,0.01
    maxepoch = 500
    
    for lam in lambs:
        for bb in bc_choice:
            batchsize = bb
            for ll in lr_choice:
                learning_rate = ll
                print('-'*50,'Finetuning','-'*50)
                print('Fingerprints:  %s'%(fp))
                print('Parameter:  lambda = %s, batchsize = %s, learning_rate = %s'%(lam,bb,ll))
                lenet = LeNetRegressor(droprate=0.2, max_epoch=maxepoch,lr=learning_rate,bc = batchsize,input_dim = input_dim,lam = lam,epsilon = epsilon)
                lenet.fit(pre_train_data, pre_train_label, lam = lam,epsilon = epsilon,verbose=False)
                results = {}

                path = "./%s/models/"%date
                a = os.path.exists( './%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate))
                if a == True:
                    print('./%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate))
                    for i in range(11):
                        number=1-(i)/20
                        LnNet_models = torch.load('./%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate), 
                                                  map_location={'cuda:1': 'cuda:1'})
                        LnNet = torch.load('./%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate))

                        y_dropout_pred = predict_reg(LnNet, pre_test_data, pre_test_label,lam)


                    results = pd.DataFrame(results,index = ['model uq_RMSE','model uq_R','model uq_R^2',
                                                            'data uq_RMSE','data uq_R','data uq_R^2',
                                                            'total uq_RMSE','total uq_R','total uq_R^2'])
#                     rmse_arr = results.iloc[3]
#                     print(rmse_arr.shape)
#                     plot_boxing_reject(list(rmse_arr)) 

                    results.T.to_csv('./%s/regression_uq_%s_lam%s_bc%s_lr%s.csv'%(date,fp,lam,batchsize,learning_rate))
                else:
                    print('./%s/models/LnNet_%s_bc%s_lr%s.pth is NAN!!!'%(date,fp,batchsize,learning_rate))
                    continue
                
#     initial_file.to_csv('./%s/3a4_880_details.csv'%date)
    test_results = pd.DataFrame(test_results,index = ['RMSE','R','R^2','model UQ','data UQ'])
    test_results.T.to_csv('./%s/regression_test.csv'%date)
    valid_results= pd.DataFrame(valid_results,index = ['train_RMSE','train_R','train_R^2','train_loss','valid_RMSE','valid_R','valid_R^2','valid_loss'])
    valid_results.T.to_csv('./%s/regression_valid.csv'%date)
            
        
        
        
        
        