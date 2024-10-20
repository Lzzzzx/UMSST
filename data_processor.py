import copy
import os
import random

import numpy as np
import torch
from torch_geometric.utils import subgraph, k_hop_subgraph
from tqdm import tqdm


class Data_processor(object):

    # 这个是测试获取数据
    @staticmethod
    def get_base_graph(adj_file, device='cpu'):
        # graph = np.load(adj_file)['adj_mx']
        graph = np.load(adj_file,allow_pickle=True)['adj_mx']  # 邻接矩阵
        graph = torch.tensor(graph, device=device, dtype=torch.float)
        # print(graph)
        # print(type(graph))
        # 处理graph转换为其他形式[[源节点编号],[目标节点编号]]
        # edge_index表示边索引 格式如上
        edge_index = []

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

        return graph, edge_index

    # 获取k-hop子图
    @staticmethod
    def get_k_hop(base_index, edge_index, k):
        #
        # edge_index 为节点列表 最好是一个节点
        # return是节点的索引列表
        subset, _, _, _ = k_hop_subgraph(node_idx=base_index, num_hops=k,
                                         edge_index=torch.LongTensor(edge_index))

        return subset

    # 观察数据信息
    @staticmethod
    def data_test(file_path):
        cat_data = np.load(file_path, allow_pickle=True, encoding='latin1')
        # print(cat_data)
        count_total = 0
        count_ma = 0
        for a, b in cat_data['y'][0][0]:
            if a > 0 or b > 0:
                count_ma += 1
            count_total += 1
        # print(count_total)
        # print(count_ma)

        # print(cat_data['x'])

        # print(cat_data['x'][0, 0])
        # print(cat_data['y'])

    # 获取最大度节点列表
    @staticmethod
    def find_max_degree(degreeofnode):
        """
            degreeofnode: 节点的度列表

            return: max_degree 最大度数
                    list_maxdegree_node_id 最大度节点列表
        """
        max_degree = max(degreeofnode)
        count = 0
        list_maxdegree_node_id = []
        for i in range(len(degreeofnode)):
            if degreeofnode[i] == max_degree:
                count += 1
                list_maxdegree_node_id.append(i)

        return max_degree, list_maxdegree_node_id


# 整体子图获取逻辑 对1个节点获取所有数据 数据的处理成本过高

def main_1(dir_path):
    adj_mx_path = os.path.join(dir_path, 'adj_mx.npz')
    train_path = os.path.join(dir_path, 'train.npz')

    # 获取到子图的信息
    data, edge_index = Data_processor.get_base_graph(adj_mx_path)

    # 获取节点个数
    graph_node_num = data.shape[0]
    # print(data.shape[0])
    # 统计各节点的度数
    degreeofnode = []
    for i in range(len(data[0])):
        count = 0
        for j in range(len(data)):
            if data[i, j] == 1:
                count += 1

        degreeofnode.append(count)
    # print(degreeofnode)

    # 统计最大度节点个数，如果超过阈值（阈值设置可以通过节点个数的百分比）

    max_count = 0  # 设置个函数处理计算最大度数节点个数
    max_count, list_maxdegree_node = Data_processor.find_max_degree(degreeofnode)

    node_selected = []  # 备选节点集合
    if max_count >= graph_node_num * 0.1:
        # 设定规则 如果最大度数相同情况过多 则直接采取随机节点策略
        node_selected.append(np.random.randint(0, high=graph_node_num, size=1))



    else:
        # 采用最大度数，随机取一个节点
        length = len(list_maxdegree_node) - 1
        index = np.random.randint(0, high=length, size=1)
        # print(index)
        node_selected.append(list_maxdegree_node[index[0]])

        pass
    # 获取
    k = 2  # k跳邻居
    sub_graph_set = Data_processor.get_k_hop(node_selected, edge_index, k)

    # 将子图输入进训练数据集当中 将子图特征数据进行筛选 剔除节点
    # print(sub_graph_set)
    base_data = np.load(train_path, allow_pickle=True, encoding='latin1')
    cat_data = base_data.copy()
    # print(cat_data)
    cat_data_x = cat_data['x']
    cat_data_y = cat_data['y']
    # 获取X数据以及Y数据

    # 这部分对x处理
    count_total = len(cat_data_x[0][0])
    # print(count_total)
    for i in range(len(cat_data_x)):  # 3000
        # 在数据集中把除图部分数据的交通流 in/out都设置为0
        count = 0
        time_length = len(cat_data_x[0])  # 数据集中时间维度的数据数量
        for j in range(time_length):
            for n in range(len(cat_data_x[i][j])):
                # 128*19
                # print(n)
                if n in sub_graph_set:
                    # print("我在里面")

                    # 在处理后的总数据当中选择交通流的和最大的top500个数据作为子图的数据 选择阈值 只要count值大于子图数量
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

    # 最后将所有子图数据都放置进训练集中（此过程不影响后续测试以及验证集的流程）

    # 流程结束


# 整体子图获取逻辑2 对节点的数据做平均处理 随机在原始数据集中抽取部分数据作为数据集增加入训练当中
def main_2(dir_path, dataset, count_node_select, k):
    """
    dir_path: 文件夹路径
    count_node_select:设置多少个节点进行增强
    k:k-hop
    """
    adj_mx_path = os.path.join(dir_path, dataset, 'adj_mx.npz')
    train_path = os.path.join(dir_path, dataset, 'train.npz')

    # 获取到子图的信息
    data, edge_index = Data_processor.get_base_graph(adj_mx_path)

    # 获取节点个数
    graph_node_num = data.shape[0]
    # print(data.shape[0])
    # 统计各节点的度数
    degreeofnode = []
    for i in range(len(data[0])):
        count = 0
        for j in range(len(data)):
            if data[i, j] == 1:
                count += 1

        degreeofnode.append(count)
    # print(degreeofnode)
    # 统计最大度节点个数，如果超过阈值（阈值设置可以通过节点个数的百分比）

    max_count = 0  # 设置个函数处理计算最大度数节点个数
    max_count, list_maxdegree_node = Data_processor.find_max_degree(degreeofnode)

    node_selected = []  # 备选节点集合
    # 备选节点个数
    count_node_select = count_node_select
    if max_count >= graph_node_num * 0.1:
        # 设定规则 如果最大度数相同情况过多 则直接采取随机节点策略
        node_selected = np.random.choice(range(0, graph_node_num), size=count_node_select)

    else:
        # 采用最大度数，随机取count_node_selected个节点
        length = len(list_maxdegree_node)
        index = np.random.choice(range(0, length), size=count_node_select)
        # print(index)
        for i in index:
            node_selected.append(list_maxdegree_node[i])

    # 获取k-hop子图列表
    k = k  # k跳邻居
    sub_graph_set = []  # 子图节点集合列表

    for i in range(len(node_selected)):
        # 筛选出节点
        sub_graph_set.append(Data_processor.get_k_hop(node_selected[i], edge_index, k))
        print(sub_graph_set)   # lzx：太久没写了 忘记逻辑了 这里的子图包含了什么是节点的id嘛



    # 将子图输入进训练数据集当中 将子图特征数据进行筛选 剔除节点
    # print(sub_graph_set)
    base_data = np.load(train_path, allow_pickle=True, encoding='latin1')
    base_data_x = base_data['x']
    base_data_y = base_data['y']
    cat_data = copy.copy(base_data)
    # print(cat_data)

    # 获取X数据以及Y数据
    cat_data_x = cat_data['x']
    cat_data_y = cat_data['y']

    # 这里是计算平均每个节点需要获取的数据量
    # print(cat_data_x)
    # print(graph_node_num)
    count_mean_node = len(cat_data_x) // graph_node_num

    # 选取部分数据这个是ndarray类型
    # 随机获取count_mean_node的数据
    for index in tqdm(range(len(sub_graph_set))):
        # print(f"loading no.{index} augmentation")
        index_rand = np.random.choice(range(0, len(cat_data_x)), size=len(sub_graph_set[index]) * count_mean_node)
        # print(len(index_rand))
        temp_list_x = []  # 暂时列表
        temp_list_y = []  # 暂时列表
        for i in index_rand:

            temp_list_x.append(cat_data_x[i])
            temp_list_y.append(cat_data_y[i])

        cat_data_x = np.array(temp_list_x)
        cat_data_y = np.array(temp_list_y)
        # print(cat_data_x)
        # print(cat_data_y)

        # 这部分对x处理
        count_total = len(cat_data_x[0][0])
        # print(count_total)
        for i in range(len(cat_data_x)):  # count_mean_node
            # 在数据集中把除图部分数据的交通流 in/out都设置为0
            count = 0
            time_length = len(cat_data_x[0])  # 数据集当中时间维度的数据数量
            for j in range(time_length):
                for n in range(len(cat_data_x[i][j])):
                    # 128*19
                    # print(n)
                    if n in sub_graph_set[index]:

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
                    if n in sub_graph_set[index]:

                        # 统计目标节点当中交通流数据不为0的个数
                        a, b = cat_data_y[i, j, n]
                        if a > 0 or b > 0:
                            count += 1
                        pass
                    else:
                        # 其余节点 就对其交通流数据设置为0
                        cat_data_y[i][j][n] = np.array([0, 0], dtype=np.float64)

        # 最后将所有子图数据都放置进训练集中（此过程不影响后续测试以及验证集的流程）
        base_data_x = np.concatenate((base_data_x, cat_data_x), axis=0)
        base_data_y = np.concatenate((base_data_y, cat_data_y), axis=0)

        # print(len(base_data_x))
        # print(len(base_data_y))
        # 流程结束

    return base_data_x, base_data_y


if __name__ == '__main__':
    # Data_processor.data_test('./data/ST-SSL_Dataset/NYCBike1/train.npz')
    # graph,edge_index=Data_processor.test('./data/ST-SSL_Dataset/NYCBike1/adj_mx.npz')
    # subset=Data_processor.get_k_hop([1,3,8],edge_index,2)
    # print(subset)
    # print(np.random.randint(0, high=128, size=1))

    # 测试部分
    adj_file='data/NYCBike1/adj_mx.npz'
    graph = np.load(adj_file,allow_pickle=True)['adj_mx']
    # print(graph)

    # main_2('./data/ST-SSL_Dataset', 'NYCBike1', 5, 2)

    # Data_processor.data_test('./data/ST-SSL_Dataset/NYCBike1/train.npz')

    # import numpy as np
    #
    # data = np.load('./data/ST-SSL_Dataset/NYCBike1/train.npz')
    # for file in data.files:
    #     print(file, data[file].shape)
