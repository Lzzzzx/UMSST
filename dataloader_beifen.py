import os
import time
import torch
import numpy as np
from tqdm import tqdm

import data_processor
from lib.utils import get_repetition_rate


def after_pretrain(model, train_data_x, graph,args):
    # best_path, model = pre_train()

    # 文件地址
    # dir_path = './data/ST-SSL_Dataset'
    # adj_path = os.path.join(dir_path, 'NYCBike1', 'adj_mx.npz')
    # train_path = os.path.join(dir_path, 'NYCBike1', 'train.npz')
    #
    # # 读取数据  调用data_processor文件
    # graph, _ = data_processor.Data_processor.get_base_graph(adj_path, 'cuda')
    # views = np.load(train_path, allow_pickle=True, encoding='latin1')['x']

    # print(model)
    # views = torch.tensor(views, device='cuda', dtype=torch.float)
    view = []

    # 这里是跑了32个 也就是规定的batchsize
    for i in range(32):
        view.append(train_data_x[i])
    view = np.array(view, dtype=np.float64)
    view = torch.tensor(view, device='cuda', dtype=torch.float)

    # if args.model=='STSSL':
    #     res = model(view, graph)
    #
    # else:
    #     res = model(view)
    res = model(view, graph)

    # 到时候要修改为输入dataloader修改后的数据  目前还没解决

    # 获取图的平均  embedding
    # model.forward()

    # print(len(res))# 32 * 1 * 128 * 64

    # print(len(res[0][0])) # 1

    # print(res.mean(dim=1))# dim=1 是求取所有元素平均值 dim=0 是取行平均值

    nodes_embedding = res.mean(dim=0)  # 1* 128*64

    # print(len(nodes_embedding))
    # print(len(nodes_embedding[0]))
    # print(len(nodes_embedding[0][0]))


    return nodes_embedding  # 32个图的各个节点的平均embedding


def get_node_topn(nodes_embedding, n, k, graph, direct=False):
    graph_average_embedding = nodes_embedding[0].mean(dim=0)
    # print(len(graph_average_embedding))

    node_scores = {}
    for i in range(len(nodes_embedding[0])):
        node_scores[i] = torch.dot(nodes_embedding[0][i], graph_average_embedding)

    # print(node_scores)

    # 获取 k数量的节点 搜索score前k的节点
    sorted_scores = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

    # print(sorted_scores)
    # 遍历前k个作为节点编号输出 随后处理数据集
    """
        分两个情况：1. 允许直接获取top n
                2. 禁止直接获取top n
    """

    """
        step1 直接提取top n
    """
    if direct:
        augmentation_top_n=[]
        for i in range(n):
            augmentation_top_n.append(sorted_scores[i][0])

        # print(augmentation_top_n)
        """
                这一块主要是为了能进行对hop-k子图节点获取 获取edge_index数据
                ---------------------------------------
        """
        # 源节点列表
        base_index = []

        # 目标节点列表
        target_index = []
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                if graph[i, j] == 1:
                    base_index.append(i)
                    target_index.append(j)

        edge_index = [base_index, target_index]
        # print(edge_index)

        second_hop_set=[]

        # 进行k-hop子图处理
        for i in augmentation_top_n:
            # print(i)
            second_hop_set.append(data_processor.Data_processor.get_k_hop(i, edge_index, k))
            """
                这里需要对子图的获取进行重复节点剔除处理
            """
            # print(second_hop_set)

    else:
        """
            step2 按评分依次添加子图序列
        """
        augmentation_all = [item[0] for item in sorted_scores[:]]
        # print(augmentation_all)
        # augmentation_top_n.append(sorted_scores[i][0])
        # 源节点列表
        base_index = []

        # 目标节点列表
        target_index = []
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                if graph[i, j] == 1:
                    base_index.append(i)
                    target_index.append(j)

        edge_index = [base_index, target_index]
        # print(edge_index)

        second_hop_set=[]
        augmentation_top_n=[]
        # 进行k-hop子图处理
        for i in augmentation_all:
            # print(i)
            # 开始剔除 每加入一个就与前面的子图检查一次，如果重复率很高那就舍弃
            if len(second_hop_set) >= n:
                break

            this_set=data_processor.Data_processor.get_k_hop(i, edge_index, k)
            access=True
            for j in range(len(second_hop_set)):
                # print(this_set)
                # print(get_repetition_rate(second_hop_set[j] ,this_set))
                if get_repetition_rate(second_hop_set[j] ,this_set)>=0.2:
                    # print(get_repetition_rate(second_hop_set[j] ,this_set))
                    access=False
                    break
            if access:
                second_hop_set.append(this_set)
                augmentation_top_n.append(i)

            """
                这里需要对子图的获取进行重复节点剔除处理
            """
            # print(second_hop_set)
            # print(augmentation_top_n)




            # print(second_hop_set)


    return graph_average_embedding,augmentation_top_n,second_hop_set


def step2_data_get(model,train_data_x,train_data_y, graph, n, k, direct,args):
    nodes_embedding = after_pretrain(model, train_data_x, graph,args)

    # 获取到topn节点以及其子图编号
    """
    lzx: 1. 需要增加一个部分用于防止子图之间节点重复的情况，
         2. 然后n修改为百分比操作(按理说应该n越多效果会越好一点，但可能会达到一定阈值)
    """
    _, _, second_hop_set = get_node_topn(nodes_embedding, n, k, graph, direct)

    # 进行数据增强操作流程 放入dataloader3进行加载
    cat_data_x, cat_data_y = train_data_x, train_data_y
    count_mean_node = len(train_data_x) // len(nodes_embedding[0])
    # print(f"count_mean_node={count_mean_node}")

    for index in tqdm(range(len(second_hop_set))):
        # print(f"loading no.{index} augmentation")
        index_rand = np.random.choice(range(0, len(cat_data_x)), size=len(second_hop_set[index]) * count_mean_node)
        # print(index_rand)
        # print(len(index_rand))
        # print("a")
        temp_list_x = []  # 暂时列表
        temp_list_y = []  # 暂时列表
        for i in index_rand:
            temp_list_x.append(cat_data_x[i])
            # print(cat_data_x[i])
            temp_list_y.append(cat_data_y[i])
            # print("b")

        cat_data_x = np.array(temp_list_x)
        cat_data_y = np.array(temp_list_y)
        # print(cat_data_x)
        # print(cat_data_y)

        # 这部分对x处理
        count_total = len(cat_data_x[0][0])

        # print(len(cat_data_x))

        # print(count_total)
        for i in range(len(cat_data_x)):  # count_mean_node
            # 在数据集中把除图部分数据的交通流 in/out都设置为0
            count = 0
            time_length = len(cat_data_x[0])  # 数据集当中时间维度的数据数量
            # print(f"time_length={time_length}")
            for j in range(time_length):
                for n in range(len(cat_data_x[i][j])):
                    # 128*19
                    # print(n)
                    if n in second_hop_set[index]:

                        # 统计目标节点当中交通流数据不为0的个数
                        a, b = cat_data_x[i, j, n]
                        if a > 0 or b > 0:
                            count += 1
                        pass
                    else:
                        # 其余节点 就对其交通流数据设置为0
                        cat_data_x[i][j][n] = np.array([0, 0], dtype=np.float64)

        # print(cat_data_x[0][0][0])
        # print(count)

        # 对y做同样的操作
        # 这部分对y处理
        count_total = len(cat_data_y[0][0])
        # print(count_total)
        for i in range(len(cat_data_y)):  # count_mean_node
            # 在数据集中把除图部分数据的交通流 in/out都设置为0
            count = 0
            time_length = len(cat_data_y[0])  # 数据集当中时间维度的数据数量
            for j in range(time_length):
                for n in range(len(cat_data_y[i][j])):
                    # 128*19
                    # print(n)
                    if n in second_hop_set[index]:

                        # 统计目标节点当中交通流数据不为0的个数
                        a, b = cat_data_y[i, j, n]
                        if a > 0 or b > 0:
                            count += 1
                        pass
                    else:
                        # 其余节点 就对其交通流数据设置为0
                        cat_data_y[i][j][n] = np.array([0, 0], dtype=np.float64)

        if index == 0:
            train_data_x = cat_data_x
            train_data_y = cat_data_y
        else:
            train_data_x = np.concatenate((train_data_x, cat_data_x), axis=0)
            train_data_y = np.concatenate((train_data_y, cat_data_y), axis=0)
    # print(train_data_x,train_data_y)
    return train_data_x,train_data_y





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
def get_dataloader_1(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        if category == 'train':
            data['x_train'], data['y_train'] = data_processor.main_2(data_dir, dataset, count_node_select=5, k=2)
        else:
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
    return dataloader, data['x_train'], data['y_train']


# 第二次预训练 加载
def get_dataloader_2(args,data_dir, dataset, batch_size, test_batch_size, model, graph, scalar_type='Standard', n=1.5, k=2):
    data = {}
    for category in ['train', 'val', 'test']:
        print(os.path.join(data_dir, dataset, category + '.npz'))

        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'), allow_pickle=True, encoding='latin1')

        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # 这个scaler的作用是什么？？？ standard scale是进行数据归一化操作的 ，可以直接将数据加入train部分，不用关系这里的concatenate函数

    # 加入增强部分的数据
    '''
        7.30 lzx: 百分比替换了
    '''
    # print(args.num_nodes)
    n = int(np.ceil(n*0.01*int(args.num_nodes)))
    print(f"n={n}")
    data['x_train'], data['y_train'] = step2_data_get(model, data['x_train'], data['y_train'], graph, n, k,args.direct,args)

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
    return dataloader,data['x_train'], data['y_train']


# 下游任务训练（不进行数据增强）
def get_dataloader_3(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
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
    loader = get_dataloader_1('../data/', 'NYCBike1', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)

    for batch_idx, (a, b) in enumerate(loader['train']):
        print(batch_idx)
        print()
        print(len(a))
        print(len(b))