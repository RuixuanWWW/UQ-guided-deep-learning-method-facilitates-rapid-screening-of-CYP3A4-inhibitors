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
import time
import os

date = time.strftime('%Y%m%d_1_lam_add',time.localtime(time.time()))
# date = '20230614_1'



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
mkdir('./%s/calibration/'%date)



device = ('cuda' )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class LeNet(nn.Module):
    def __init__(self, droprate=0.5,test=0):
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
        self.final2= torch.nn.Linear(10, 3)
        self.final1= torch.nn.Linear(10, 1)
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
        out_probability = torch.abs(self.final2(out_value))
        out = torch.cat((self.final1(out_value), out_probability), dim=1)
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


def get_Rmse(data,label):
    mse=0
    for i in range(data.shape[0]):
        mse+=(data[i]-label[i])**2
    return math.sqrt(mse/data.shape[0])


def plot_show_uncertainty(data_un,model_un):
    """
    :param x: 输入数据
    :return num_bin: 条柱数目
    """
    data_un=pd.DataFrame(data_un)
    data_static = data_un.describe().T
    print(data_static)
    model_un = pd.DataFrame(model_un)
    model_static = model_un.describe().T
    print(model_static)

    plt.figure()
    # plt.xlim(x_min, x_max)
    plt.hist(data_un, bins=50,color= 'blue', linewidth=0.5, alpha=0.5, label="data_uncertainty")##根据情况画把 如果隔得很远分开画
    plt.hist(model_un, bins=3, color= 'red',linewidth=0.6,alpha=1,label="model_uncertainty")#
    
    plt.ylabel('Sample number')
    plt.xlabel('value')
    plt.legend()
    plt.show()
    
def predict_reg(model, x, label,lam,T=10):###原无label
    X = Variable(x.type(torch.FloatTensor).to(device))
    model.eval() ####
#     print(X.shape)##############
    output = model(X.unsqueeze(1)).data.cpu()#.numpy()
    label = label.cpu().numpy()

    
    min_val = 1e-6
    # Split the outputs into the four distribution parameters
    mu, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
    v = torch.nn.Softplus()(loglambdas) + min_val
    alpha = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
    beta = torch.nn.Softplus()(logbetas) + min_val
    
    mu= mu.squeeze(1).numpy()
    v = v.squeeze(1).numpy()
    alpha = alpha.squeeze(1).numpy()
    beta = beta.squeeze(1).numpy()
    
    data_uncertainty= beta/(alpha-1)
    model_uncertainty=beta/(alpha-1)/v
    total_uncertainty = data_uncertainty + model_uncertainty
#     plot_show_uncertainty(data_uncertainty,model_uncertainty)
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
#         print(number)
#         print(m)
#         print(len(mu[choose_sign]),uncertainty.shape[0],)

#         np.save('./%s/backups/predlabs_%s_bc%s_lr%s_%s_%s.npy'%(date,fp,batchsize,learning_rate,type,number),mu[choose_sign])
#         np.save('./%s/backups/truelabs_%s_bc%s_lr%s_%s_%s.npy'%(date,fp,batchsize,learning_rate,type,number),label[choose_sign])
#         np.save('./%s/backups/uq_%s_bc%s_lr%s_%s_%s.npy'%(date,fp,batchsize,learning_rate,type,number),uncertainty[choose_sign])
        
        if m == 1:
            np.save('./%s/backups/%.2f_receive_data_uncertainty.npy'%(date,number),uncertainty[choose_sign])


        if number == 1 and m ==0:
            test_result.append(Rmse)
            test_result.append(reject_r)
            test_result.append(reject_R_squre)
            test_result.append(np.mean(uncertainty))
        if number == 1 and m ==1:
            test_result.append(np.mean(uncertainty))
#             print('!!! para_result',para_result)
#             print(batchsize,learning_rate)
#         else:
#             print('error')
            test_results['fp:%s,lam:%s,bc: %s,lr: %s'%(fp,lam,batchsize,learning_rate)] = test_result
        
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
    

    global number,maxepoch,input_dim #,batchsize,learning_rate
    input_dim = len(pre_train_data[0])
    number=1
    lambs = [0.15,0.2,0.25,0.3]#0.005,0.01,0.05,0.1,0.2,0.4,0.6,1,2]#,0.01,0.05,0.1,0.2,0.4,0.6,1,2]
    epsilon = 1e-4
    bc_choice = [16,32,64,128,256,512]#,32,64,128,256,512]#16,,51216,,64,128,256,64,128
    lr_choice = [0.01,0.001,0.0001]#,0.01,0.0001,,0.01
    maxepoch = 300
    
    for lam in lambs:
        for bb in bc_choice:
            batchsize = bb
            for ll in lr_choice:
                learning_rate = ll
                print('-'*50,'Finetuning','-'*50)
                print('Fingerprints:  %s'%(fp))
                
                print('Parameter:  lambda = %s, batchsize = %s, learning_rate = %s'%(lam,bb,ll))
                lenet = LeNetRegressor(droprate=0.0, max_epoch=maxepoch,lr=learning_rate,bc = batchsize,input_dim = input_dim,lam = lam,epsilon = epsilon)
                lenet.fit(pre_train_data, pre_train_label, lam = lam,epsilon = epsilon,verbose=False)
                results = {}

                path = "./%s/models/"%date
                a = os.path.exists('./%s/models/LnNet_%s_lam%s_bc%s_lr%s.pth'%(date,fp,lam,batchsize,learning_rate))
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
                    results.T.to_csv('./%s/regression_uq_%s_lam%s_bc%s_lr%s.csv'%(date,fp,lam,batchsize,learning_rate))
                else:
                    print('./%s/models/LnNet_%s_bc%s_lr%s.pth is NAN!!!'%(date,fp,batchsize,learning_rate))
                    continue
    test_results = pd.DataFrame(test_results,index = ['RMSE','R','R^2','model UQ','data UQ'])
    test_results.T.to_csv('./%s/regression_test.csv'%date)
    valid_results= pd.DataFrame(valid_results,index = ['train_RMSE','train_R','train_R^2','train_loss','valid_RMSE','valid_R','valid_R^2','valid_loss'])
    valid_results.T.to_csv('./%s/regression_valid.csv'%date)
            
        
        
        
        
        