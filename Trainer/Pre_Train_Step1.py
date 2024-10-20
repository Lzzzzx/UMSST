import argparse
from copy import deepcopy

import yaml
from tqdm import tqdm

import data_processor
import lib
from lib.Params_predictor import get_predictor_params
from lib.dataloader import Get_Dataloader_Pretrain_Step2, Get_Dataloader_STssl, StandardScaler
from model.layers import (STEncoder, Align, TemporalConvLayer, MLP)
import os
import time
import numpy as np
import torch
import torch.nn as nn
import traceback

from lib.attenuation import (
    Attenuation
)

from lib.logger import (
    get_logger,
    PD_Stats,
)
from lib.utils import (
    get_log_dir,
    get_model_params,
    dwa,
    init_seed,
    get_model_params,
    load_graph, masked_mae_loss,
)
from lib.metrics import test_metrics

'''
    这段是为baseline的loss做准备
'''

def MAE_torch(pred, true, mask_value=None):
    # print(pred)
    # print(true)
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    # print(mae_loss)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    # print(torch.mean(mae_loss,dim=0))

    return torch.mean(torch.abs(true - pred)), torch.abs(true - pred)

def huber_loss(pred, true, mask_value=None, delta=1.0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    residual = torch.abs(pred - true)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res)), None
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)
def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        preds = scaler.inverse_transform(preds)
        # print(preds.shape)

        labels = scaler.inverse_transform(labels)
        # print(preds.shape)
        # print(labels.shape)

        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        # print(mae,mae_loss)
        return mae, mae_loss
    return loss

def scaler_huber_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        preds = scaler.inverse_transform(preds)
        labels = scaler.inverse_transform(labels)
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

'''
    结束· ·   ·   ··········
'''

class Pre_trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args,args_other=None):
        super(Pre_trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.args = args
        self.args_other=args_other

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_encoder_model.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch):
        self.model.train()

        # 测试用
        # count=0

        total_loss = 0
        total_sep_loss = np.zeros(3)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # count+=1
            # print(f"第{count}次")
            self.optimizer.zero_grad()

            # input shape: n,l,v,c; graph shape: v,v;

            # TODO 这里加一个属性用来分别是否需要加衰减函数
            if self.args.set_attenuation:
                # 将衰减放于此处进行
                if self.args.attenuation == 'log':
                    data = Attenuation.func_attenuation_log_fast(data, self.args.batch_size, self.args.attenuation_rate)


                elif self.args.attenuation == 'exp':
                    data = Attenuation.func_attenuation_exp_fast(data, self.args.batch_size, self.args.attenuation_rate)


                elif self.args.attenuation == 'inv':
                    data = Attenuation.func_attenuation_inv_fast(data, self.args.batch_size, self.args.attenuation_rate)


                else:
                    print("attenuation is out of ['log','exp','inv']")

            # print("test")
            # print(data.shape)
            '''暂时不需要 '''

            # if self.args.model=='STSSL':
            #     repr = self.model(data, self.graph)  # nvc
            #     loss, sep_loss = self.model.loss(repr, target, self.scaler)
            #     # print(repr[0].shape)
            # else:
            #     mean_data = data.mean()
            #     std_data = data.std()
            #     scaler_data = StandardScaler(mean_data, std_data)
            #     label = target
            #     # print(data.shape)
            #     repr = self.model(data)
            #     # print(repr.shape)
            #     # print(target.shape)
            #     if self.args_other.loss_func == 'mask_mae':
            #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
            #         # print('============================scaler_mae_loss')
            #     elif self.args_other.loss_func == 'mask_huber':
            #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
            #         # print('============================scaler_mae_loss')
            #         # print(args.model, Mode)
            #     '''
            #         这里要考虑一下repr是32,19,128,2  label是32,1,128,2但是他们的总数相同
            #
            #     '''
            #
            #     loss, _ = loss(repr, label)

            repr = self.model(data, self.graph)  # nvc
            loss, sep_loss = self.model.loss(repr, target, self.scaler)

            assert not torch.isnan(loss)
            loss.backward()

            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            if self.args.model=='STSSL':
                total_sep_loss += sep_loss
            else:
                total_sep_loss=None

        train_epoch_loss = total_loss / self.train_per_epoch
        # total_sep_loss = total_sep_loss / self.train_per_epoch

        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # 衰减函数添加部分
                # TODO 这里加一个属性用来分别是否需要加衰减函数
                if self.args.set_attenuation:
                    # 这里不知需不需要添加衰减函数
                    if self.args.attenuation == 'log':
                        # print("采用的是log衰减")
                        data = Attenuation.func_attenuation_log_fast(data, self.args.batch_size, self.args.attenuation_rate)


                    elif self.args.attenuation == 'exp':
                        data = Attenuation.func_attenuation_exp_fast(data, self.args.batch_size, self.args.attenuation_rate)


                    elif self.args.attenuation == 'inv':
                        data = Attenuation.func_attenuation_inv_fast(data, self.args.batch_size, self.args.attenuation_rate)


                    else:
                        print("attenuation is out of ['log','exp','inv']")


                '''暂时不需要'''
                # if self.args.model == 'STSSL':
                #     repr = self.model(data, self.graph)  # nvc
                #     loss, sep_loss = self.model.loss(repr, target, self.scaler)
                # else:
                #     mean_data = data.mean()
                #     std_data = data.std()
                #     scaler_data = StandardScaler(mean_data, std_data)
                #     label = target[..., :2]
                #     # print(data.shape)
                #     repr = self.model(data)
                #     # print(target.shape)
                #     if self.args_other.loss_func == 'mask_mae':
                #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                #         # print('============================scaler_mae_loss')
                #     elif self.args_other.loss_func == 'mask_huber':
                #         loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                #         # print('============================scaler_mae_loss')
                #         # print(args.model, Mode)
                #     '''
                #         这里要考虑一下repr是32,19,128,2  label是32,1,128,2但是他们的总数相同
                #
                #     '''
                #     loss, _ = loss(repr, label)

                repr = self.model(data, self.graph)  # nvc
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
            if self.args.model=="STSSL":
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

                repr = model(data, graph)
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


class ST_pre(nn.Module):
    def __init__(self, args):
        super(ST_pre, self).__init__()
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model // 2), args.d_model],
                                                     [args.d_model, int(args.d_model // 2), args.d_model]],
                                 input_length=args.input_length, num_nodes=args.num_nodes,
                                 batch_size=args.batch_size,attenuation=args.attenuation,attenuation_rate=args.attenuation_rate,
                                 droprate=args.dropout)

        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)

        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args

    def forward(self, view, graph):
        # print(f"view.size={view.size()}")
        repr, repr_downstream = self.encoder(view, graph)  # view1: n,l,v,c; graph: v,v
        # print(f"repr.size={repr.size()}")
        return repr

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
        # print(f"z.size={z.size()}")
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


# pretrain



def St_ssl_encoder_train(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    ## load dataset

    # 修改下 dataloader 由于以前的思路是需要进行一次数据增强 但现在不需要这样的操作 所以需要对其做一个对比实验 观测一下这个是否会有比较大的影响
    dataloader = Get_Dataloader_STssl(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    # dataloader, train_data_x, train_data_y = get_dataloader_3(
    #     data_dir=args.data_dir,
    #     dataset=args.dataset,
    #     batch_size=args.batch_size,
    #     test_batch_size=args.test_batch_size,
    # )


    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    # print(args.num_nodes)
    model_args=None
    '''
        现在不需要这一步骤，后续需要保证前两个阶段信息都是stssl的encoder部分，在第三个阶段再进行baseline模型的训练
    '''
    # 这里修改一下 如果模型是其他模型则需要对其他模型进行初始化
    # if args.model=='STFGNN':
    #     model_args=get_predictor_params(args)
    #     from model.STFGNN.STFGNN import STFGNN
    #     # dim_in = 2
    #     model = STFGNN(model_args)
    #     model = model.to(args.device)
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)
    # elif args.model == 'STGCN':
    #     model_args=get_predictor_params(args)
    #     from model.STGCN.stgcn import STGCN
    #     model = STGCN(model_args, args.device)
    #     model = model.to(args.device)
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)
    # elif args.model=='STSGCN':
    #     model_args = get_predictor_params(args)
    #     from model.STSGCN.STSGCN import STSGCN
    #     model = STSGCN(model_args, args.device)
    #     model = model.to(args.device)
    #     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
    #                                  weight_decay=0, amsgrad=False)

    # else:
    #     ## init model and set optimizer
    #     model = ST_pre(args).to(args.device)
    #     model_parameters = get_model_params([model])
    #     optimizer = torch.optim.Adam(
    #         params=model_parameters,
    #         lr=args.lr_init,
    #         eps=1.0e-8,
    #         weight_decay=0,
    #         amsgrad=False
    #     )

    # init model and set optimizer
    model = ST_pre(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters,
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )

    ## start training
    trainer = Pre_trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=graph,
        args=args,
        args_other=model_args
    )

    # 最佳模型地址
    best_path = trainer.best_path
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
    # return results, best_path, model, train_data_x, train_data_y, graph
    return results, best_path, model, graph






# 第一次预训练
def Pretrain_Step1():
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

    # 原版本
    # _, best_path, model, graph, train_data_x, train_data_y = Encoder_train(args)
    # return best_path, model, graph, train_data_x, train_data_y

    # 第一次预训练不添加数据增强
    _, best_path, model, graph = St_ssl_encoder_train(args)
    return best_path, model, graph




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
    #                     type=str, help='the configuration to use')
    # args = parser.parse_args()
    #
    # print(f'Starting experiment with configurations in {args.config_filename}...')
    # time.sleep(3)
    # configs = yaml.load(
    #     open(args.config_filename),
    #     Loader=yaml.FullLoader
    # )
    #
    # args = argparse.Namespace(**configs)
    # # 训练encoder效果
    # _,best_path=Encoder_train(args)


    # 原版本
    # _, pre_model, _ , _ , _ =pre_train()

    # 不添加数据增强预训练
    _, pre_model, _ = Pretrain_Step1()


    # step2_pretrain_again(train_data_x,train_data_y,20,2)
    # best_path, model, train_data_x, train_data_y, graph=Step2_pretrain(pre_model)

    # print(best_path)



