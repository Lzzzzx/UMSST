import os
import time
import torch 
import numpy as np

import data_processor


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

def STDataloader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    # print('{} scalar is used!!!'.format(scalar_type))
    # time.sleep(3)
    return scalar


# 预训练dataloader （进行了数据增强）
def Get_Dataloader_Pretrain_Step2(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        if category=='train':
            data['x_train'],data['y_train']=data_processor.main_2(data_dir,dataset,count_node_select=5,k=2)
        else:
        # 加入
            print(os.path.join(data_dir, dataset, category + '.npz'))

            cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'),allow_pickle=True,encoding='latin1')

            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']

    # 这个scaler的作用是什么？？？ standard scale是进行数据归一化操作的 ，可以直接将数据加入train部分，不用关系这里的concatenate函数

    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])


    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'], 
        data['y_train'], 
        batch_size, 
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'], 
        data['y_val'], 
        test_batch_size, 
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'], 
        data['y_test'], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader

# 第二次预训练 加载
def Get_Dataloader_Pretrain_step2(data_dir, dataset, batch_size, test_batch_size, model, graph, scalar_type='Standard', n=20, k=2):
    data = {}
    for category in ['train', 'val', 'test']:


        print(os.path.join(data_dir, dataset, category + '.npz'))

        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'), allow_pickle=True, encoding='latin1')

        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # 这个scaler的作用是什么？？？ standard scale是进行数据归一化操作的 ，可以直接将数据加入train部分，不用关系这里的concatenate函数

    # 加入增强部分的数据
    data['x_train'], data['y_train'] = step2_data_get(model,data['x_train'], data['y_train'],graph, n, k)

    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'],
        data['y_train'],
        batch_size,
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'],
        data['y_val'],
        test_batch_size,
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'],
        data['y_test'],
        test_batch_size,
        shuffle=False,
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader



# 下游任务训练（不进行数据增强）
def Get_Dataloader_STssl(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        # 加入
        print(os.path.join(data_dir, dataset, category + '.npz'))

        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'), allow_pickle=True, encoding='latin1')

        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # 这个scaler的作用是什么？？？ standard scale是进行数据归一化操作的 ，可以直接将数据加入train部分，不用关系这里的concatenate函数

    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'],
        data['y_train'],
        batch_size,
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'],
        data['y_val'],
        test_batch_size,
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'],
        data['y_test'],
        test_batch_size,
        shuffle=False,
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader




if __name__ == '__main__':
    loader = Get_Dataloader_Pretrain_Step2('../data/', 'NYCBike1', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)


    for batch_idx, (a,b) in enumerate(loader['train']):
        print(batch_idx)
        print()
        print(len(a))
        print(len(b))