import os
import time
import numpy as np
import torch
from copy import deepcopy
import yaml
import argparse
import traceback
import warnings

from lib.Params_predictor import get_predictor_params

warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
sys.path.append('../..')

from lib.logger import (
    get_logger,
    PD_Stats,
)
from lib.utils import (
    get_log_dir,
    dwa,
    init_seed,
    get_model_params,
    load_graph,
)
from lib.metrics import test_metrics
from Trainer.Pre_Train_Step1 import Pretrain_Step1, scaler_mae_loss
from Trainer.Pre_Train_Step2 import Pretrain_Step2
from model.models import STSSL
from lib.dataloader import Get_Dataloader_STssl, StandardScaler


class ST_SSL_Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args, args_other, pre_model=None):
        super(ST_SSL_Trainer, self).__init__()
        self.pretrain_model = pre_model
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
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch, loss_weights=None):
        self.model.train()

        total_loss = 0
        total_sep_loss = np.zeros(3)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # input shape: n,l,v,c; graph shape: v,v;
            # repr1, repr2 = self.model(data, self.graph)  # nvc
            # loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)

            if self.args.model == 'STSSL':
                repr1, repr2 = self.model(data, self.graph)  # nvc
                loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)
            else:
                if self.pretrain_model==None:
                    repr = self.model(data)
                    # mean_data = data.mean()
                    # std_data = data.std()
                    scaler_data = self.scaler
                    label = target[..., :2]
                else:
                    _, repr_downstream = self.pretrain_model(data, self.graph)
                    # mean_data = repr_downstream.mean()
                    # std_data = repr_downstream.std()
                    # scaler_data = StandardScaler(mean_data, std_data)
                    scaler_data=self.scaler
                    label = target[..., :2]
                    # print(repr_downstream.shape)
                    repr = self.model(repr_downstream)

                if self.args_other.loss_func == 'mask_mae':
                    loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                    # print('============================scaler_mae_loss')
                elif self.args_other.loss_func == 'mask_huber':
                    loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                    # print('============================scaler_mae_loss')

                loss, _ = loss(repr, label)

            assert not torch.isnan(loss)
            loss.backward()

            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            # total_sep_loss += sep_loss
            if self.args.model == 'STSSL':
                total_sep_loss += sep_loss
            else:
                total_sep_loss = None

        train_epoch_loss = total_loss / self.train_per_epoch
        # total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader, loss_weights=None):
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # repr1, repr2 = self.model(data, self.graph)
                # loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)
                if self.args.model == 'STSSL':
                    repr1, repr2 = self.model(data, self.graph)  # nvc
                    loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)
                else:
                    if self.pretrain_model == None:
                        repr = self.model(data)
                        # mean_data = data.mean()
                        # std_data = data.std()
                        # scaler_data = StandardScaler(mean_data, std_data)
                        scaler_data=self.scaler
                        label = target[..., :2]
                    else:
                        _, repr_downstream = self.pretrain_model(data, self.graph)
                        # mean_data = repr_downstream.mean()
                        # std_data = repr_downstream.std()
                        scaler_data = self.scaler
                        label = target[..., :2]

                        repr = self.model(repr_downstream)

                    if self.args_other.loss_func == 'mask_mae':
                        loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                        # print('============================scaler_mae_loss')
                    elif self.args_other.loss_func == 'mask_huber':
                        loss = scaler_mae_loss(scaler_data, mask_value=0.001)
                        # print('============================scaler_mae_loss')

                    loss, _ = loss(repr, label)

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
            # self.logger.info('loss weights: {}'.format(loss_weights))
            if self.args.model == "STSSL":
                train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            else:
                train_epoch_loss, loss_t = self.train_epoch(epoch)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            if self.args.model=="STSSL":

                val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)
            else:
                val_epoch_loss = self.val_epoch(epoch,val_dataloader)
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
        if self.args.model=='STSSL':
            test_results = self.test(self.model, self.test_loader, self.scaler,
                                     self.graph, self.logger, self.args)
        else:
            test_results = self.test(self.model, self.test_loader, self.scaler,
                                     self.graph, self.logger, self.args, self.pretrain_model)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }
        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args, pretrain_model=None):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # repr1, repr2 = model(data, graph)
                # pred_output = model.predict(repr1, repr2)
                #
                # y_true.append(target)
                # y_pred.append(pred_output)
                if args.model=='STSSL':
                    repr1, repr2 = model(data, graph)
                    pred_output = model.predict(repr1, repr2)

                    y_true.append(target)
                    y_pred.append(pred_output)
                else:
                    if pretrain_model == None:
                        repr = model(data)
                    else:

                        _, repr_downstream = pretrain_model(data, graph)
                        repr = model(repr_downstream)
                    y_true.append(target)
                    y_pred.append(repr)
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


def St_ssl_supervisor(args, pre_train_model_path=None):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    ## load dataset
    dataloader = Get_Dataloader_STssl(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    model_args = None
    model_pretrain = None
    if args.model == 'STFGNN':
        if pre_train_model_path == None:
            model_args = get_predictor_params(args)
            from model.STFGNN.STFGNN import STFGNN
            dim_in = 2
            # print("1")
            model = STFGNN(model_args, dim_in)
            model = model.to(args.device)
        else:
            model_args = get_predictor_params(args)
            from model.STFGNN.STFGNN import STFGNN
            dim_in = 64

            from Trainer.Pre_Train_Step2 import ST_Premodel
            model_pretrain = ST_Premodel(args).to(args.device)
            # 载入预训练部分的encoder 与 mlp参数

            # 载入预训练模型
            print("路径为:{}".format(pre_train_model_path))
            arg_encoder_total = torch.load(pre_train_model_path)['model']

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
            model_pretrain.mlp.load_state_dict(arg_mlp)
            model_pretrain.encoder.load_state_dict(arg_encoder)

            model = STFGNN(model_args, dim_in)
            model = model.to(args.device)
            print("路径为:{}".format(pre_train_model_path))
            # arg_encoder_total = torch.load(pre_train_model_path)['model']
            # model.load_state_dict(arg_encoder_total)

        print("已成功载入")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=0, amsgrad=False)
    elif args.model == 'STGCN':
        if pre_train_model_path == None:
            model_args = get_predictor_params(args)
            from model.STGCN.stgcn import STGCN
            dim_in = 2
            dim_out = 2
            model = STGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)
        else:
            model_args = get_predictor_params(args)
            from model.STGCN.stgcn import STGCN
            dim_in = 64
            dim_out = 2

            from Trainer.Pre_Train_Step2 import ST_Premodel
            model_pretrain = ST_Premodel(args).to(args.device)
            # 载入预训练部分的encoder 与 mlp参数

            # 载入预训练模型
            print("路径为:{}".format(pre_train_model_path))
            arg_encoder_total = torch.load(pre_train_model_path)['model']

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
            model_pretrain.mlp.load_state_dict(arg_mlp)
            model_pretrain.encoder.load_state_dict(arg_encoder)

            model = STGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)

        print("已成功载入")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=0, amsgrad=False)
    elif args.model == 'STSGCN':
        if pre_train_model_path == None:
            model_args=get_predictor_params(args)
            from model.STSGCN.STSGCN import STSGCN
            dim_in = 2
            dim_out = 2
            model = STSGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)
        else:
            model_args = get_predictor_params(args)
            from model.STSGCN.STSGCN import STSGCN
            dim_in = 64
            dim_out = 2
            from Trainer.Pre_Train_Step2 import ST_Premodel
            model_pretrain = ST_Premodel(args).to(args.device)
            # 载入预训练部分的encoder 与 mlp参数

            # 载入预训练模型
            print("路径为:{}".format(pre_train_model_path))
            arg_encoder_total = torch.load(pre_train_model_path)['model']

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
            model_pretrain.mlp.load_state_dict(arg_mlp)
            model_pretrain.encoder.load_state_dict(arg_encoder)

            model = STSGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)
            print("路径为:{}".format(pre_train_model_path))

        print("已成功载入")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=0, amsgrad=False)

    elif args.model == 'GMAN':
        if pre_train_model_path == None:
            model_args = get_predictor_params(args)
            from model.STSGCN.STSGCN import STSGCN
            dim_in = 2
            dim_out = 2
            model = STSGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)
        else:
            model_args = get_predictor_params(args)
            from model.STSGCN.STSGCN import STSGCN
            dim_in = 64
            dim_out = 2
            from Trainer.Pre_Train_Step2 import ST_Premodel
            model_pretrain = ST_Premodel(args).to(args.device)
            # 载入预训练部分的encoder 与 mlp参数

            # 载入预训练模型
            print("路径为:{}".format(pre_train_model_path))
            arg_encoder_total = torch.load(pre_train_model_path)['model']

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
            model_pretrain.mlp.load_state_dict(arg_mlp)
            model_pretrain.encoder.load_state_dict(arg_encoder)

            model = STSGCN(model_args, args.device, dim_in, dim_out)
            model = model.to(args.device)
            print("路径为:{}".format(pre_train_model_path))

        print("已成功载入")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=0, amsgrad=False)

    else:
        ## init model and set optimizer


        if args.train_mode=='base':
            model = STSSL(args).to(args.device)
        elif args.train_mode=='pre_train':
            model = STSSL(args).to(args.device)
            # 载入预训练部分的encoder 与 mlp参数
            # 载入预训练模型
            print("路径为:{}".format(pre_train_model_path))
            arg_encoder_total = torch.load(pre_train_model_path)['model']

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

        model_parameters = get_model_params([model])
        optimizer = torch.optim.Adam(
            params=model_parameters,
            lr=args.lr_init,
            eps=1.0e-8,
            weight_decay=0,
            amsgrad=False
        )

    ## start training
    if args.model=='STSSL':
        trainer = ST_SSL_Trainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            graph=graph,
            args=args,
            args_other=model_args
        )
    else:
        trainer = ST_SSL_Trainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            graph=graph,
            args=args,
            args_other=model_args,
            pre_model=model_pretrain
        )
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
    return results



if __name__ == '__main__':
    # print(torch.cuda.device)

    # # 第一步预训练
    # pre_path, pre_model, _ = Pretrain_Step1()
    #
    # # 第二步预训练
    # # 从pre_train 当中获取到train_data的数据
    # best_path, model, train_data_x, train_data_y, graph = Pretrain_Step2(pre_path, pre_model, k=3, n=1) # n为top  k为hop

    # UMSST
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='../configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    parser.add_argument('-model', default='STSSL', type=str, help='training model')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(1)
    configs = yaml.load(
        open(args.config_filename),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    results=St_ssl_supervisor(args)
    # model_supervisor(args, pre_path)
