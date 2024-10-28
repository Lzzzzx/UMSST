import argparse
import os
import time
import traceback
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch import nn

import data_processor
from dataloader_beifen import get_dataloader_2
from lib.Params_predictor import get_predictor_params
from lib.dataloader import StandardScaler
from lib.logger import get_logger, PD_Stats
from lib.metrics import test_metrics

from lib.utils import init_seed, load_graph, get_model_params, get_log_dir, dwa, masked_mae_loss
from model.layers import STEncoder, MLP
from Trainer.Pre_Train_Step1 import Pretrain_Step1, scaler_mae_loss


class Pre_Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args, args_other):
        super(Pre_Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.args = args
        self.args_other = args_other

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)

        self.best_path = os.path.join(self.args.log_dir, 'best_encoder_model.pth')
        self.step_2_encoder_path = os.path.join(self.args.step_2_encoder_dir, self.args.dataset,
                                                f'{self.args.step_1}_{self.args.step_2}_{args.attenuation}_{args.attenuation_rate}_encoder_step2.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_sep_loss = np.zeros(3)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            '''这部分暂时不需要'''
            # input shape: n,l,v,c; graph shape: v,v;
            # if self.args.model == 'STSSL':
            #     repr, repr_downstream = self.model(data, self.graph)  # nvc
            #     loss, sep_loss = self.model.loss(repr, target, self.scaler)
            # else:
            #     mean_data = data.mean()
            #     std_data = data.std()
            #     scaler_data = StandardScaler(mean_data, std_data)
            #     label = target
            #     # print(data.shape)
            #     repr = self.model(data)
            #     # print(target.shape)
            #     if self.args_other.loss_func == 'mask_mae':
            #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
            #         # print('============================scaler_mae_loss')
            #     elif self.args_other.loss_func == 'mask_huber':
            #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
            #         # print('============================scaler_mae_loss')
            #     loss, _ = loss(repr, label)

            repr, repr_downstream = self.model(data, self.graph)  # nvc
            loss, sep_loss = self.model.loss(repr, target, self.scaler)

            # 这里做个简单的调试 如果碰到nan就直接跳过 而不是做断言错误爆出
            if torch.isnan(loss):
                continue

            loss.backward()

            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            if self.args.model == 'STSSL':
                total_sep_loss += sep_loss
            else:
                total_sep_loss = None

        train_epoch_loss = total_loss / self.train_per_epoch
        # total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # if self.args.model == 'STSSL':
                #     repr = self.model(data, self.graph)  # nvc
                #     loss, sep_loss = self.model.loss(repr, target, self.scaler)
                # else:
                #     mean_data = data.mean()
                #     std_data = data.std()
                #     scaler_data = StandardScaler(mean_data, std_data)
                #     label = target[..., :2]
                #
                #     repr = self.model(data)
                #
                #     if self.args_other.loss_func == 'mask_mae':
                #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                #         # print('============================scaler_mae_loss')
                #     elif self.args_other.loss_func == 'mask_huber':
                #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                #         # print('============================scaler_mae_loss')
                #
                #     loss, _ = loss(repr, label)

                repr, repr_downstream = self.model(data, self.graph)  # nvc
                loss, sep_loss = self.model.loss(repr, target, self.scaler)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3)  # (1.0, 1.0, 1.0)
        for epoch in range(1, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks
            if self.args.model == "STSSL":
                if self.args.use_dwa:
                    loss_tm2 = loss_tm1
                    loss_tm1 = loss_t
                    if (epoch == 1) or (epoch == 2):
                        loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                    else:
                        loss_weights = dwa(loss_tm1, loss_tm2, self.args.temp)
                # self.logger.info('loss weights: {}'.format(loss_weights))
            else:
                pass
            train_epoch_loss, loss_t = self.train_epoch(epoch)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            if not self.args.debug:
                self.training_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))

                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args.early_stop_patience))
                break

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                         "Total training time: {:.2f} min\t"
                         "best loss: {:.4f}\t"
                         "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))


        if os.path.isdir(os.path.join(self.args.step_2_encoder_dir, self.args.dataset)) == False:
            os.makedirs(os.path.join(self.args.step_2_encoder_dir, self.args.dataset), exist_ok=True)
        print('step_2 pretrain best encoder mode saved to {}'.format(self.step_2_encoder_path))

        torch.save(save_dict, self.step_2_encoder_path)


        # test
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler,
                                 self.graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }
        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # repr = model(data, graph)
                # pred_output = model.predict(repr)
                #
                # y_true.append(target)
                # y_pred.append(pred_output)
                # if args.model=='STSSL':
                #
                #     repr = model(data, graph)
                #     pred_output = model.predict(repr)
                #
                #     y_true.append(target)
                #     y_pred.append(pred_output)
                # else:
                #     repr = model(data)
                #     y_true.append(target)
                #     y_pred.append(repr)
                repr, repr_downstream = model(data, graph)
                pred_output = model.predict(repr)

                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])
        # outflow
        mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])

        return np.stack(test_results, axis=0)


class ST_Premodel(nn.Module):
    def __init__(self, args):
        super(ST_Premodel, self).__init__()
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model // 2), args.d_model],
                                                     [args.d_model, int(args.d_model // 2), args.d_model]],
                                 input_length=args.input_length, num_nodes=args.num_nodes,
                                 batch_size=args.batch_size, attenuation=args.attenuation,
                                 attenuation_rate=args.attenuation_rate,
                                 droprate=args.dropout)

        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)

        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args

    def forward(self, view, graph):
        # print(view)
        repr, repr_downstream = self.encoder(view, graph)  # view1: n,l,v,c; graph: v,v
        # print(len(repr))
        return repr, repr_downstream

    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()

    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def predict(self, z):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        return self.mlp(z)

    def loss(self, z, y_true, scaler):
        l = self.pred_loss(z, y_true, scaler)
        sep_loss = [l.item()]
        loss = l

        return loss, sep_loss

    def pred_loss(self, z, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(z))
        y_true = scaler.inverse_transform(y_true)

        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
               (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss

    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)


def Step2_encoder_train(args, pre_model_path, pre_model, n=1.5, k=2):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    model_args = None
    ## load dataset
    dataloader, train_data_x, train_data_y = get_dataloader_2(
        args=args,
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        model=pre_model,
        graph=graph,
        test_batch_size=args.test_batch_size,
        n=n,
        k=k
    )
    # if args.model=='STFGNN':
    #     model_args=get_predictor_params(args)
    #     from model.STFGNN.STFGNN import STFGNN
    #     dim_in = 2
    #     model = STFGNN(model_args, dim_in)
    #     model = model.to(args.device)
    #     print("路径为:{}".format(pre_model_path))
    #     arg_encoder_total = torch.load(pre_model_path)['model']
    #     model.load_state_dict(arg_encoder_total)
    #
    #     print("已成功载入")
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)
    # elif args.model == 'STGCN':
    #     model_args=get_predictor_params(args)
    #     from model.STGCN.stgcn import STGCN
    #     dim_in = 2
    #     dim_out = 2
    #     model = STGCN(model_args, args.device, dim_in, dim_out)
    #     model = model.to(args.device)
    #     print("路径为:{}".format(pre_model_path))
    #     arg_encoder_total = torch.load(pre_model_path)['model']
    #     model.load_state_dict(arg_encoder_total)
    #
    #     print("已成功载入")
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)
    #
    # elif args.model == 'STSGCN':
    #     model_args=get_predictor_params(args)
    #     from model.STSGCN.STSGCN import STSGCN
    #     dim_in = 2
    #     dim_out = 2
    #     model = STSGCN(model_args, args.device, dim_in, dim_out)
    #     model = model.to(args.device)
    #     print("路径为:{}".format(pre_model_path))
    #     arg_encoder_total = torch.load(pre_model_path)['model']
    #     model.load_state_dict(arg_encoder_total)
    #
    #     print("已成功载入")
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)
    # else:
    #     ## init model and set optimizer
    #     model = ST_Premodel(args).to(args.device)
    #     model_parameters = get_model_params([model])
    #     optimizer = torch.optim.Adam(
    #         params=model_parameters,
    #         lr=args.lr_init,
    #         eps=1.0e-8,
    #         weight_decay=0,
    #         amsgrad=False
    #     )
    #     # 载入之前的模型参数
    #     # 载入预训练模型
    #     print("路径为:{}".format(pre_model_path))
    #     arg_encoder_total = torch.load(pre_model_path)['model']
    #     # model.load
    #     print("已成功载入")
    #
    #     key_to_mlp = list(arg_encoder_total.keys())[-4:]
    #     key_to_encoder = list(arg_encoder_total.keys())[:-4]
    #     # 获取encoder部分模型参数
    #     arg_encoder = deepcopy(arg_encoder_total)
    #     for key in key_to_mlp:
    #         del arg_encoder[key]
    #     # 获取mlp部分模型参数
    #     arg_mlp = deepcopy(arg_encoder_total)
    #     for key in key_to_encoder:
    #         del arg_mlp[key]
    #
    #     # 删除mlp.与encoder.前缀
    #     for key in key_to_encoder:
    #         sp = key.split("encoder.")[1]
    #         arg_encoder[sp] = arg_encoder.pop(key)
    #
    #     for key in key_to_mlp:
    #         sp = key.split("mlp.")[1]
    #         arg_mlp[sp] = arg_mlp.pop(key)
    #
    #     # 载入参数
    #     model.mlp.load_state_dict(arg_mlp)
    #     model.encoder.load_state_dict(arg_encoder)
    #
    #     # optimizer = torch.optim.Adam(
    #     #     params=model_parameters,
    #     #     lr=args.lr_init,
    #     #     eps=1.0e-8,
    #     #     weight_decay=0,
    #     #     amsgrad=False
    #     # )

    ## init model and set optimizer
    model = ST_Premodel(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters,
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )
    # 载入之前的模型参数
    # 载入预训练模型
    print("路径为:{}".format(pre_model_path))
    arg_encoder_total = torch.load(pre_model_path)['model']
    # model.load
    print("已成功载入")

    key_to_mlp = list(arg_encoder_total.keys())[-4:]
    key_to_encoder = list(arg_encoder_total.keys())[:-4]
    # 获取encoder部分模型参数
    arg_encoder = deepcopy(arg_encoder_total)
    for key in key_to_mlp:
        del arg_encoder[key]
    # 获取mlp部分模型参数
    arg_mlp = deepcopy(arg_encoder_total)
    for key in key_to_encoder:
        del arg_mlp[key]

    # 删除mlp.与encoder.前缀
    for key in key_to_encoder:
        sp = key.split("encoder.")[1]
        arg_encoder[sp] = arg_encoder.pop(key)

    for key in key_to_mlp:
        sp = key.split("mlp.")[1]
        arg_mlp[sp] = arg_mlp.pop(key)

    # 载入参数
    model.mlp.load_state_dict(arg_mlp)
    model.encoder.load_state_dict(arg_encoder)

    # optimizer = torch.optim.Adam(
    #     params=model_parameters,
    #     lr=args.lr_init,
    #     eps=1.0e-8,
    #     weight_decay=0,
    #     amsgrad=False
    # )

    ## start training
    trainer = Pre_Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=graph,
        args=args,
        args_other=model_args
    )

    # 最佳模型地址
    # best_path = trainer.best_path
    best_path = trainer.step_2_encoder_path
    print(best_path)

    results = None
    try:
        if args.mode == 'train':
            results = trainer.train()  # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                                   graph, trainer.logger, trainer.args)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())

    return results, best_path, model, train_data_x, train_data_y, graph


# 预训练后的流程 （数据增强再次部分）
def after_pretrain(model, train_data_x, graph):
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

    # 这里是跑了32个 也就是规定的batchsize  这里或许可以优化调整

    for i in range(32):
        view.append(train_data_x[i])
    view = np.array(view, dtype=np.float64)
    view = torch.tensor(view, device='cuda', dtype=torch.float)

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


def get_node_topn(nodes_embedding, n, k, graph):
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
    augmentation_top_k = []
    for i in range(n):
        augmentation_top_k.append(sorted_scores[i][0])

    # print(augmentation_top_k)
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
    """
        --------------到这里-------------------
    """

    second_hop_set = []

    # 进行k-hop子图处理
    for i in augmentation_top_k:
        second_hop_set.append(data_processor.k_hop_subgraph(i, k, edge_index))

    # print(second_hop_set)

    return graph_average_embedding, augmentation_top_k, second_hop_set


# 获取数据流程 输入给dataloader2部分进行dataloader的加载 为第二次预训练做准备
def step2_data_get(model, train_data_x, train_data_y, graph, n, k):
    nodes_embedding = after_pretrain(model, train_data_x, graph)

    # 获取到topn节点以及其子图编号
    _, _, second_hop_set = get_node_topn(nodes_embedding, n, k, graph)

    # 进行数据增强操作流程 放入dataloader3进行加载
    cat_data_x, cat_data_y = train_data_x, train_data_y
    count_mean_node = len(train_data_x) // len(nodes_embedding)

    # for index in tqdm(range(len(second_hop_set))):
    for index in range(len(second_hop_set)):
        # print(f"loading no.{index} augmentation")
        index_rand = np.random.choice(range(0, len(cat_data_x)), size=len(second_hop_set[index]) * count_mean_node)
        # print('a')
        # print(len(index_rand))
        temp_list_x = []  # 暂时列表
        temp_list_y = []  # 暂时列表
        for i in index_rand:
            if i in index_rand:
                # print('b')
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
                    if n in second_hop_set[index]:
                        # print("我在里面")

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
                        # print("我在里面")

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
    return train_data_x, train_data_y


def Pretrain_Step2(pre_path, pre_model, n, k):
    # 再次进行一轮预训练
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/NYCTaxi.yaml',
                        type=str, help='the configuration to use')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    # 训练encoder效果
    _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args, pre_model_path=pre_path,
                                                                                 pre_model=pre_model, n=n, k=k)

    return best_path, model, train_data_x, train_data_y, graph


if __name__ == '__main__':
    pre_path, pre_model, _, _, _ = Pretrain_Step1()

    best_path, model, train_data_x, train_data_y, graph = Pretrain_Step2(pre_path, pre_model, k=2, n=20)

    print(best_path)
