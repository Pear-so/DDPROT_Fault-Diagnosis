# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
#from datasets.SequenceDatasets import dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
from scipy.signal import resample
class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

class Normalize(object):
    def __init__(self, type = "mean-std"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


class CWRU_GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, list_data,transform=None):
        self.data = list_data['data'].tolist()
        self.label = list_data['label'].tolist()
        self.transform = transform

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        
        return data, labels
    def __len__(self):
        return len(self.data)
normlizetype="0-1"
transform = transforms.Compose([
        
        Reshape(),
                Normalize(normlizetype),
                Retype(),])
#Digital data was collected at 12,000 samples per second
signal_size = 2560



dataname= {0:["K004_09_07_10.csv","KA04_09_07_10.csv",  "KA16_09_07_10.csv", "KB24_09_07_10.csv",  "KI18_09_07_10.csv", "KI21_09_07_10.csv"],  # 900/7/10
            1:["K004_15_01_10.csv","KA04_15_01_10.csv",  "KA16_15_01_10.csv", "KB24_15_01_10.csv",  "KI18_15_01_10.csv", "KI21_15_01_10.csv"],  # 1500/1/10
            2:["K004_15_07_04.csv","KA04_15_07_04.csv",  "KA16_15_07_04.csv", "KB24_15_07_04.csv",  "KI18_15_07_04.csv", "KI21_15_07_04.csv"],  # 1500/7/4
            3:["K004_15_07_10.csv","KA04_15_07_10.csv",  "KA16_15_07_10.csv", "KB24_15_07_10.csv",  "KI18_15_07_10.csv", "KI21_15_07_10.csv"]}  # 1500/7/10


label = [0,1,2,3,4,5]
label2 = [0,1,2,3,4,5]
label3 = [0,1,2,3,4,5]
label4 = [0,1,2,3,4,5]
root = r'C:\Users\fuche\Desktop\PU'


###############


####################

#####构建client1
def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = np.loadtxt(filename,encoding='utf8')
    fl = fl[2000:642000]
    fl = fl.reshape(-1,1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
        
    # while end <= fl.shape[0]:
    #     x = fl[start:end]
    #     x = np.fft.fft(x)
    #     x = np.abs(x) / len(x)
    #     x = x[range(int(x.shape[0] / 2))]
    #     x = x.reshape(-1,1)
    #     data.append(x)
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size

    return data, lab

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root,dataname[N[k]][i])
            data1, lab1 = data_load(path1,label=label[i])
            data += data1
            lab +=lab1

    return [data, lab]


########构建client2
def data_load_2(filename, label2):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = np.loadtxt(filename,encoding='utf8')
    fl = fl[2000:642000]
    fl = fl.reshape(-1,1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label2)
        start += signal_size
        end += signal_size
        
        
    # while end <= fl.shape[0]:
    #     x = fl[start:end]
    #     x = np.fft.fft(x)
    #     x = np.abs(x) / len(x)
    #     x = x[range(int(x.shape[0] / 2))]
    #     x = x.reshape(-1,1)
    #     data.append(x)
    #     lab.append(label2)
    #     start += signal_size
    #     end += signal_size

    return data, lab


def get_files_2(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root,dataname[N[k]][i])
            data1, lab1 = data_load_2(path1,label2=label2[i])
            data += data1
            lab +=lab1

    return [data, lab]


########构建client3
def data_load_3(filename, label3):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = np.loadtxt(filename,encoding='utf8')
    fl = fl[2000:642000]
    fl = fl.reshape(-1,1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
       # x = x.reshape(50,50,1)
        data.append(x)
        lab.append(label3)
        start += signal_size
        end += signal_size
        
        
    # while end <= fl.shape[0]:
    #     x = fl[start:end]
    #     x = np.fft.fft(x)
    #     x = np.abs(x) / len(x)
    #     x = x[range(int(x.shape[0] / 2))]
    #     x = x.reshape(-1,1)
    #     data.append(x)
    #     lab.append(label3)
    #     start += signal_size
    #     end += signal_size

    return data, lab


def get_files_3(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root,dataname[N[k]][i])
            data1, lab1 = data_load_3(path1,label3=label3[i])
            data += data1
            lab +=lab1

    return [data, lab]

######构建client4

def data_load_4(filename, label4):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = np.loadtxt(filename,encoding='utf8')
    fl = fl[2000:642000]
    fl = fl.reshape(-1,1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        #x = x.reshape(50, 50,1)
        data.append(x)
        lab.append(label4)
        start += signal_size
        end += signal_size
        
    # while end <= fl.shape[0]:
    #     x = fl[start:end]
    #     x = np.fft.fft(x)
    #     x = np.abs(x) / len(x)
    #     x = x[range(int(x.shape[0] / 2))]
    #     x = x.reshape(-1,1)
    #     data.append(x)
    #     lab.append(label4)
    #     start += signal_size
    #     end += signal_size

    return data, lab


def get_files_4(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root,dataname[N[k]][i])
            data1, lab1 = data_load_4(path1,label4=label4[i])
            data += data1
            lab +=lab1

    return [data, lab]



def load_client_1():

    list_data_T1 = get_files(root, [0])
    data_pd = pd.DataFrame({"data": list_data_T1[0], "label": list_data_T1[1]})
    train_client_1_data_pd, test_client_1_data_pd = train_test_split(data_pd, test_size=0.2, 
                                            random_state=40, stratify=data_pd["label"])
    
    train_client_1_data_pd = train_client_1_data_pd[0:1184]
    # train_client_1_data_pd = train_client_1_data_pd[0:992]
    # test_client_1_data_pd = test_client_1_data_pd[0:1000]
    return train_client_1_data_pd, test_client_1_data_pd


def load_client_2():

    list_data_T2 = get_files_2(root, [1])
    data_pd = pd.DataFrame({"data": list_data_T2[0], "label": list_data_T2[1]})
    train_client_2_data_pd, test_client_2_data_pd = train_test_split(data_pd, test_size=0.2, 
                                            random_state=40, stratify=data_pd["label"])
    
    
    train_client_2_data_pd = train_client_2_data_pd[0:1184]
    # test_client_2_data_pd = test_client_2_data_pd[0:1000]
    return train_client_2_data_pd, test_client_2_data_pd


def load_client_3():
    list_data_T3 = get_files_3(root, [2])
    data_pd = pd.DataFrame({"data": list_data_T3[0], "label": list_data_T3[1]})
    train_client_3_data_pd, test_client_3_data_pd = train_test_split(data_pd, test_size=0.2, 
                                            random_state=40, stratify=data_pd["label"])
    
    
    train_client_3_data_pd = train_client_3_data_pd[0:1184]
    
    # test_client_3_data_pd = test_client_3_data_pd[0:1000]

    return train_client_3_data_pd, test_client_3_data_pd

def load_client_4():

    list_data_T4 = get_files_4(root, [3])
    data_pd = pd.DataFrame({"data": list_data_T4[0], "label": list_data_T4[1]})
    train_client_4_data_pd, test_client_4_data_pd = train_test_split(data_pd, test_size=0.2, 
                                            random_state=40, stratify=data_pd["label"])
    
    
    train_client_4_data_pd = train_client_4_data_pd[0:1184]
    
    # test_client_4_data_pd = test_client_4_data_pd[0:1000]
    return train_client_4_data_pd, test_client_4_data_pd

def PU_dataset_read(domain, batch_size):
    if domain == "client_1":
        train_image_pd,  test_image_pd  = load_client_1()
    elif domain == "client_2":
        train_image_pd,  test_image_pd  = load_client_2()
    elif domain == "client_3":
        train_image_pd,  test_image_pd  = load_client_3()
    elif domain == "client_4":
        train_image_pd,  test_image_pd = load_client_4()

    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function

    # raise train and test data loader
    train_dataset = CWRU_GetLoader(train_image_pd, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = CWRU_GetLoader(test_image_pd, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader