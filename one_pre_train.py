import argparse
from copy import deepcopy

import yaml
from tqdm import tqdm

import data_processor
import lib
from lib.dataloader import get_dataloader_1, get_dataloader_3
from model.layers import (STEncoder, Align, TemporalConvLayer, MLP)
import os
import time
import numpy as np
import torch
import torch.nn as nn
import traceback

from new_trainer import model_supervisor
from pre_train import Encoder_train
from step_2_pretrain import Step2_pretrain, Step2_encoder_train
import pandas as pd


class Train_all(object):

    # 初始化需要循环迭代的参数
    def __init__(self, step_1, step_2, dataset):
        super().__init__()
        # 取5个常量种子
        self.dataset = dataset
        # self.seed = [31, 42, 53, 64, 75]  # [36,10,15,31, 42, 53, 64, 75,99,87,123]
        self.seed = [15, 31, 75, 87, 53]  # bike2
        self.k = [1, 2, 3, 4]  # hop  1  2  3  4
        self.n = [1, 2, 4, 8]  # top  1  2  4  8
        self.args = None
        self.results = []
        self.step_1 = step_1  # 这个代表第step组参数的实验 [k,n]
        self.step_2 = step_2
        self.att = {'log': [0.5, 1, 2],
                    'exp': [0.01, 0.001, 0.0001],
                    'inv': [0.5, 1, 2]}

    def get_seed(self, step):
        # 获取随机种子
        return self.seed[step]

    # 通过这里跑整体流程  最好有个参数继承的关系 这样不需要一直加载模型的参数 而可以1阶段流程跑完
    def train_all(self):

        # 加载参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_filename', default=f'configs/{self.dataset}.yaml',
                            type=str, help='the configuration to use')
        args = parser.parse_args()

        print(f'Starting experiment with configurations in {args.config_filename}...')
        time.sleep(1)

        configs = yaml.load(
            open(args.config_filename),
            Loader=yaml.FullLoader
        )
        args = argparse.Namespace(**configs)
        if args.set_attenuation:
            print("进入了衰减函数训练流程")
            for i in range(9):
                if i // 3 == 0:
                    args.attenuation = 'log'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                elif i // 3 == 1:
                    args.attenuation = 'exp'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                elif i // 3 == 2:
                    args.attenuation = 'inv'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                print(f"训练的衰减函数为{args.attenuation},其参数值为{args.attenuation_rate}")
                # 跑衰减系数版本
                for j in range(len(self.seed)):
                    print(f"第{j+1}个随机种子，值为{self.seed[j]}")
                    args.seed = self.seed[j]
                    self.args = args
                    # 第一步预训练
                    _, pre_path, pre_model, _ = Encoder_train(args)

                    # 第二步预训练
                    # 从pre_train 当中获取到train_data的数据
                    # 训练encoder效果
                    _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args,
                                                                                                 pre_model_path=pre_path,
                                                                                                 pre_model=pre_model,
                                                                                                 n=self.n[self.step_1],
                                                                                                 k=self.k[self.step_2])

                    result = model_supervisor(args, best_path)
                    self.results.append(result)
                    print(result['test_results'])

                # 保存其均值与方差
                self.save_data_att()

        # 需要跑五次
        else:
            print("未进入衰减函数训练流程。")
            for j in range(len(self.seed)):

                print(f"第{j + 1}个随机种子，值为{self.seed[j]}")
                args.seed = self.seed[j]
                self.args = args
                # 第一步预训练
                _, pre_path, pre_model, _ = Encoder_train(args)

                # 第二步预训练
                # 从pre_train 当中获取到train_data的数据
                # 训练encoder效果
                _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args,
                                                                                             pre_model_path=pre_path,
                                                                                             pre_model=pre_model,
                                                                                             n=self.n[self.step_1],
                                                                                             k=self.k[self.step_2])

                result = model_supervisor(args, best_path)
                self.results.append(result)
                print(result['test_results'])

            # 保存其均值与方差 加上每个种子的结果
            self.save_data()

    def get_results(self):
        end_results = []
        for i in range(len(self.seed)):
            end_results.append(self.results[i]['test_results'])
        print(end_results)
        # 数据含义为[[in_MAE,in_MAPE],[out_MAE,out_MAPE]]

        return end_results

    # 获取结果均值和方差
    def get_averange(self):
        results = self.get_results()
        results_mean = np.mean(results, axis=0)
        results_std = np.std(results, axis=0)
        return results, results_mean, results_std

    # 保存均值与方差于res文件夹下
    def save_data(self):
        results, result_mean, result_std = self.get_averange()
        save_data_avg = pd.DataFrame({'in_MAE_mean': [result_mean[0][0]],
                                      'out_MAE_mean': [result_mean[1][0]],
                                      'in_MAPE_mean': [result_mean[0][1]],
                                      'out_MAPE_mean': [result_mean[1][1]],
                                      'in_MAE_std': [result_std[0][0]],
                                      'out_MAE_std': [result_std[1][0]],
                                      'in_MAPE_std': [result_std[0][1]],
                                      'out_MAPE_std': [result_std[1][1]],
                                      })
        in_mae = []
        out_mae = []
        in_mape = []
        out_mape = []
        for result in results:
            in_mae.append(result[0][0])
            out_mae.append(result[1][0])
            in_mape.append(result[0][1])
            out_mape.append(result[1][1])
        save_data_every = pd.DataFrame({'seed': self.seed,
                                        'in_MAE': in_mae,
                                        'out_MAE': out_mae,
                                        'in_MAPE': in_mape,
                                        'out_MAPE': out_mape
                                        }).set_index('seed')

        # 保存平均
        save_data_avg.to_csv(f"res/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_avg.csv")
        print(f'已经保存至 res/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_avg.csv文件中')

        # 保存每次的结果
        save_data_every.to_csv(f"res/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_every.csv")
        print(f'已经保存至 res/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_every.csv文件中')
        # 初始化结果字典
        self.results = []
        print("results已经初始化")

    def save_data_att(self):
        results, result_mean, result_std = self.get_averange()
        save_data_avg = pd.DataFrame({'in_MAE_mean': [result_mean[0][0]],
                                      'out_MAE_mean': [result_mean[1][0]],
                                      'in_MAPE_mean': [result_mean[0][1]],
                                      'out_MAPE_mean': [result_mean[1][1]],
                                      'in_MAE_std': [result_std[0][0]],
                                      'out_MAE_std': [result_std[1][0]],
                                      'in_MAPE_std': [result_std[0][1]],
                                      'out_MAPE_std': [result_std[1][1]],
                                      })
        in_mae = []
        out_mae = []
        in_mape = []
        out_mape = []
        for result in results:
            in_mae.append(result[0][0])
            out_mae.append(result[1][0])
            in_mape.append(result[0][1])
            out_mape.append(result[1][1])
        save_data_every = pd.DataFrame({'seed': self.seed,
                                        'in_MAE': in_mae,
                                        'out_MAE': out_mae,
                                        'in_MAPE': in_mape,
                                        'out_MAPE': out_mape
                                        }).set_index('seed')

        # 保存平均
        save_data_avg.to_csv(
            f"res_att/{self.args.dataset}_k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.attenuation}_{self.args.attenuation_rate}_avg.csv")
        print(
            f'已经保存至 res_att/{self.args.dataset}_k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.attenuation}_{self.args.attenuation_rate}_avg.csv文件中')

        # 保存每次的结果
        save_data_every.to_csv(f"res_att/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.attenuation}{self.args.attenuation_rate}_every.csv")
        print(f'已经保存至 res_att/{self.args.dataset}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_\
        {self.args.attenuation}{self.args.attenuation_rate}_every.csv文件中')
        # 初始化结果字典
        self.results=[]
        print("results已经初始化")



if __name__ == '__main__':
    # 测试一下
    # trainer = Train_all(0, 0, "BJTaxi")  # 这个输入的参数代表训练第几组参数的实验 [k,n]
    # trainer.train_all()

    step_1 = 3
    print(f"step_1={step_1}")

    for step_2 in range(4):
        print(f"--------------这 是 第 {step_2} 组 实 验.-----------------")

        # 初始化训练器 dataset BJTaxi NYCBike1 NYCBike2 NYCTaxi
        trainer = Train_all(step_1, step_2, "NYCBike2")  # 这个输入的参数代表训练第几组参数的实验 [k,n]
        trainer.train_all()
        # result_mean, result_std = trainer.get_averange()
        # print(f"均值为{result_mean},方差为{result_std}") #数据含义为[[in_MAE,in_MAPE],[out_MAE,out_MAPE]]
        # trainer.save_data()
        print(f"---------第{step_2}组实验结束--------------")
        print()

    # 成功就用来保存
