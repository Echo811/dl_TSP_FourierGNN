from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import datetime
import numpy as np
import pandas as pd

# traffic data
class Dataset_Dhfm(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # load_data = np.load(root_path, allow_pickle=True)  # 源码
        # data = load_data.transpose()  # 源码
        '''
            2023年12月3日  16点21分  修改失败，修改后运行结果不合理
            疑问：    date列是如何处理的？
                    和FreTS模型处理方式的区别是什么？
            修改思路：源码中读取的是npy文件，但是数据集未提供，于是找其他项目上相同的traffic.csv数据集
                    文件类型不一样，导致报错
                    
                    于是尝试通过：data = pd.read_csv(root_path).iloc[:, 1:].to_numpy()
                    即：和其他类的处理方式一样，结果是可以读出数据集，可以正确划分数据集，但是运行结果异常
                    
                    【运行结果】
                    /usr/local/Anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:160: RuntimeWarning: overflow encountered in reduce
                      ret = umr_sum(arr, axis, dtype, out, keepdims)
                    RAW : MAPE 170.283874156%; MAE 157765612669704276809875456.000000000; RMSE     inf.
                    | end of epoch   0 | time: 234.76s | train_total_loss 0.0035 | val_loss   inf
                    ...
                    
            后续思路：
                    查看该项目是如何处理date列的【该项目基于GNN，那么更要看懂数据处理的新方法】
                        
        '''
        # data = pd.read_csv(root_path).iloc[:, 1:].to_numpy()
        # print(data.shape)

        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data) * self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

# ECG dataset
class Dataset_ECG(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path)

        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        data = np.array(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*self.train_ratio)
            end = int(len(data)*(self.val_ratio+self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*(self.val_ratio+self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        files = os.listdir(root_path)
        solar_data = []
        time_data = []
        for file in files:
            if not os.path.isdir(file):
                if file.startswith('DA_'):
                    data = pd.read_csv(root_path + '\\' + file).values
                    raw_time = data[:, 0:1]
                    if time_data == []:
                        time_data = raw_time
                    raw_data = data[:, 1:data.shape[1]]
                    raw_data = raw_data.transpose()
                    solar_data.append(raw_data)
        solar_data = np.array(solar_data).squeeze(1).transpose()
        time_data = np.array(time_data)
        out = np.concatenate((time_data, solar_data), axis=1)
        self.data = []
        for item in out:
            tmp = item[0]
            dt = datetime.datetime.strptime(tmp, "%m/%d/%y %H:%M")
            if dt.hour >=8 and dt.hour <= 17:
                self.data.append(item[1:out.shape[1]-1])

        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
        self.data = mms.fit_transform(self.data)
        if self.flag == 'train':
            begin = 0
            end = int(len(self.data)*self.train_ratio)
            self.trainData = self.data[begin:end]
            self.train_nextData = self.data[begin:end]
        if self.flag == 'val':
            begin = int(len(self.data)*self.train_ratio)
            end = int(len(self.data)*(self.train_ratio+self.val_ratio))
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end]
        if self.flag == 'test':
            begin = int(len(self.data)*(self.train_ratio+self.val_ratio))
            end = len(self.data)
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

class Dataset_Wiki(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path).values
        raw_data = data[:, 1:data.shape[1]]
        df = pd.DataFrame(raw_data)
        # data cleaning
        self.data = df.dropna(axis=0, how='any').values.transpose()
        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            self.data = mms.fit_transform(self.data)
        if self.flag == 'train':
            begin = 0
            end = int(len(self.data)*self.train_ratio)
            self.trainData = self.data[begin:end]
            self.train_nextData = self.data[begin:end]
        if self.flag == 'val':
            begin = int(len(self.data)*self.train_ratio)
            end = int(len(self.data)*(self.train_ratio+self.val_ratio))
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end]
        if self.flag == 'test':
            begin = int(len(self.data)*(self.train_ratio+self.val_ratio))
            end = len(self.data)
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end]

    def __getitem__(self, index):
        # data timestamp
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        # return the time data , next time data and time
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len
