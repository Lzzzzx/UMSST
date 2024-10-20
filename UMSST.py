import argparse
import os

import yaml

import time
import numpy as np

from Trainer.ST_SSL_Train import St_ssl_supervisor
from Trainer.Pre_Train_Step1 import St_ssl_encoder_train
from Trainer.Pre_Train_Step2 import Step2_encoder_train
import pandas as pd


class UMSST(object):
    # 初始化需要循环迭代的参数
    def __init__(self):
        super().__init__()
        # self.dataset = dataset
        # self.seed = [31, 42, 53, 64, 75]  # [36,10,15,31, 42, 53, 64, 75,99,87,123]
        self.step_1 = None
        self.step_2 = None
        self.set_attenuation = None
        self.seed = [15, 31, 75, 87, 53]  # bike2
        self.k = [1, 2, 3, 4]  # hop  1  2  3  4
        self.n = [1, 1.5, 2, 2.5]  # top结果
        self.args = None
        self.results = []
        self.model=None
        # self.step_1 = step_1  # 这个代表第step组参数的实验 [k,n]
        # self.step_2 = step_2
        self.att = {'log': [0.5, 1, 2],
                    'exp': [0.01, 0.001, 0.0001],
                    'inv': [0.5, 1, 2]}

    def get_seed(self, step):
        # 获取随机种子
        return self.seed[step]

    # 通过这里跑整体流程  最好有个参数继承的关系 这样不需要一直加载模型的参数 而可以1阶段流程跑完
    def train_all(self,args):
        # 加载参数
        # parser = argparse.ArgumentParser()
        # parser.add_argument('-model', default='STSSL', type=str, help='training model')
        # parser.add_argument('--config_filename', default=f'configs/NYCBike1.yaml',
        #                     type=str, help='the configuration to use')
        # # parser.add_argument('--dataset',default=f'NYCBike1', help='dataset')
        # parser.add_argument('--step_1',default='0',type=int, help='the group of k,from 0 to 3, each means [1, 2, 3, 4] ')
        # parser.add_argument('--step_2', default='0', type=int, help='the group of n,from 0 to 3,each means [1, 1.5, 2, 2.5]')
        # parser.add_argument('--set_attenuation', default=False, type=eval,help='whether to use the attenuation function,False or True')
        #
        # args1 = parser.parse_args()
        #
        # print(f'Starting experiment with configurations in {args1.config_filename}...')
        #
        # # print(args1.set_attenuation)
        # self.step_1 = args1.step_1
        # self.step_2 = args1.step_2
        # # self.set_attenuation = args1.set_attenuation
        # time.sleep(1)
        #
        # configs = yaml.load(
        #     open(args1.config_filename),
        #     Loader=yaml.FullLoader
        # )
        #
        # # print(args1.set_attenuation)
        # configs['set_attenuation'] = args1.set_attenuation
        #
        #
        # args = argparse.Namespace(**configs)
        # # print(args.num_nodes)
        # args.model = args1.model


        # print(args.set_attenuation)

        if args.set_attenuation:
            print("attenuation func module")
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
                print(f"attenuation function:{args.attenuation}, weight:{args.attenuation_rate}")
                for j in range(len(self.seed)):
                    print(f"No_{j+1}rand seed, value:{self.seed[j]}")
                    args.seed = self.seed[j]
                    self.args = args
                    '''前两个阶段不改变，只需要修改最后一个阶段下游任务的训练'''

                    # Pretrain_step1
                    _, pre_path, pre_model, _ = St_ssl_encoder_train(args)

                    # Pretrain_step2
                    _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args,
                                                                                                 pre_model_path=pre_path,
                                                                                                 pre_model=pre_model,
                                                                                                 n=self.n[self.step_1],
                                                                                                 k=self.k[self.step_2])

                    result = St_ssl_supervisor(args, best_path)
                    self.results.append(result)
                    print(result['test_results'])

                self.save_data_att(args)

        else:
            print("without attenuation module.")
            for j in range(len(self.seed)):

                print(f"No_{j + 1} rand seed, value{self.seed[j]}")
                args.seed = self.seed[j]
                self.args = args
                # first pretrain
                _, pre_path, pre_model, _ = St_ssl_encoder_train(args)
                # second pretrain
                _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args,
                                                                                             pre_model_path=pre_path,
                                                                                             pre_model=pre_model,
                                                                                             n=self.n[self.step_1],
                                                                                             k=self.k[self.step_2])
                # downstream train
                result = St_ssl_supervisor(args, best_path)
                print(result)
                self.results.append(result)
                print(result['test_results'])

            self.save_data(args)

    def get_results(self):
        end_results = []
        for i in range(len(self.seed)):
            end_results.append(self.results[i]['test_results'])
        print(end_results)

        return end_results

    def get_averange(self):
        """
            get mean and std
        """
        results = self.get_results()
        results_mean = np.mean(results, axis=0)
        results_std = np.std(results, axis=0)
        return results, results_mean, results_std

    # 保存均值与方差于res文件夹下
    def save_data(self,args):
        """
            save mean and std in dir res/
        """
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


        if not (os.path.exists(f'res/{self.args.dataset}/{self.args.model}') and os.path.isdir(f'res/{self.args.dataset}/{self.args.model}')):
            os.mkdir(f'res/{self.args.dataset}/{self.args.model}')

        save_avg_path = f"res/{self.args.dataset}/{self.args.model}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.train_mode}_avg.csv"
        save_every_path = f"res/{self.args.dataset}/{self.args.model}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.train_mode}_every.csv"

        save_data_avg.to_csv(save_avg_path)
        print(f'saved res/{save_avg_path}文件中')

        save_data_every.to_csv(save_every_path)
        print(f'saved res/{save_every_path}文件中')

        self.results = []
        print("results inited")

    def save_data_att(self,args):
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

        if not (os.path.exists(f'res_att/{self.args.dataset}/{self.args.model}') and os.path.isdir(f'res_att/{self.args.dataset}/{self.args.model}')):
            os.mkdir(f'res_att/{self.args.dataset}/{self.args.model}')

        save_avg_path=f"res_att/{self.args.dataset}/{self.args.model}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.attenuation}_{self.args.attenuation_rate}_{self.args.train_mode}_avg.csv"
        save_every_path=f"res_att/{self.args.dataset}/{self.args.model}/k{self.k[self.step_1]}_n{self.n[self.step_2]}_{self.args.attenuation}{self.args.attenuation_rate}_{self.args.train_mode}_every.csv"

        save_data_avg.to_csv(save_avg_path)
        print(f'saved {save_avg_path}')

        save_data_every.to_csv(save_every_path)
        print(f'saved {save_every_path}')

        # init dict
        self.results=[]
        print("results inited")

    def train_one_pre_train(self):
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
            print("Attenuation module")
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
                print(f"attenuation function:{args.attenuation}, weight:{args.attenuation_rate}")
                # attenuation func
                for j in range(len(self.seed)):
                    print(f"No_{j + 1} rand_seed, value:{self.seed[j]}")
                    args.seed = self.seed[j]
                    self.args = args
                    # first pretrain
                    _, pre_path, pre_model, _ = St_ssl_encoder_train(args)

                    result = St_ssl_supervisor(args, pre_path)
                    self.results.append(result)
                    print(result['test_results'])


                # self.save_data_att()
                self.save_data_one_pretrain_att()

        else:
            print("without attenuation module")
            for j in range(len(self.seed)):
                print(f"No_{j + 1} rand seed, value:{self.seed[j]}")
                args.seed = self.seed[j]
                self.args = args
                # 第一步预训练
                _, pre_path, pre_model, _ = St_ssl_encoder_train(args)

                result = St_ssl_supervisor(args, pre_path)
                self.results.append(result)
                print(result['test_results'])

            self.save_data_one_pretrain()

    def save_data_one_pretrain_att(self):
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
            f"res_one_train_att/{self.args.dataset}_{self.args.attenuation}_{self.args.attenuation_rate}_avg.csv")
        print(
            f'saved res_one_train_att/{self.args.dataset}_{self.args.attenuation}_{self.args.attenuation_rate}_avg.csv文件中')

        # 保存每次的结果
        save_data_every.to_csv(
            f"res_one_train_att/{self.args.dataset}/{self.args.attenuation}{self.args.attenuation_rate}_every.csv")
        print(f'saved res_one_train_att/{self.args.dataset}/{self.args.attenuation}{self.args.attenuation_rate}_every.csv文件中')
        # 初始化结果字典
        self.results = []
        print("results have inited")


    def save_data_one_pretrain(self):
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
        save_data_avg.to_csv(f"res_one_train/{self.args.dataset}/avg.csv")
        print(f'saved res_one_train/{self.args.dataset}/avg.csv文件中')

        # 保存每次的结果
        save_data_every.to_csv(f"res_one_train/{self.args.dataset}/every.csv")
        print(f'saved res_one_train/{self.args.dataset}/every.csv文件中')
        # 初始化结果字典
        self.results = []
        print("results have inited")


    def train_bjtaxi(self):
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
            print("attenuation func module")
            for i in range(7,9):
                if i // 3 == 0:
                    args.attenuation = 'log'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                elif i // 3 == 1:
                    args.attenuation = 'exp'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                elif i // 3 == 2:
                    args.attenuation = 'inv'
                    args.attenuation_rate = self.att[args.attenuation][i % 3]
                print(f"attenuation func:{args.attenuation}, weight:{args.attenuation_rate}")
                # 跑衰减系数版本
                for j in range(len(self.seed)):
                    print(f"No_{j + 1} rand seed, value:{self.seed[j]}")
                    args.seed = self.seed[j]
                    self.args = args
                    # step1
                    _, pre_path, pre_model, _ = St_ssl_encoder_train(args)

                    # step2
                    _, best_path, model, graph, train_data_x, train_data_y = Step2_encoder_train(args,
                                                                                                 pre_model_path=pre_path,
                                                                                                 pre_model=pre_model,
                                                                                                 n=self.n[self.step_1],
                                                                                                 k=self.k[self.step_2])
                    # step_3
                    result = St_ssl_supervisor(args, best_path)
                    self.results.append(result)
                    print(result['test_results'])

                # 保存其均值与方差
                self.save_data_att()

    def train_baseline_base(self, args):
        if args.set_attenuation:
            print("attenuation func module")
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
                print(f"attenuation function:{args.attenuation}, weight:{args.attenuation_rate}")
                for j in range(len(self.seed)):
                    print(f"No_{j + 1}rand seed, value:{self.seed[j]}")
                    args.seed = self.seed[j]
                    self.args = args
                    # args['pre_train'] = False
                    result = St_ssl_supervisor(args, None)
                    self.results.append(result)
                    print(result['test_results'])

                self.save_data_att(args)

        else:
            print("without attenuation module.")
            for j in range(len(self.seed)):
                print(f"No_{j + 1} rand seed, value{self.seed[j]}")
                args.seed = self.seed[j]
                self.args = args

                # downstream train
                result = St_ssl_supervisor(args, None)
                print(result)
                self.results.append(result)
                print(result['test_results'])

            self.save_data(args)


def main_total(step_1):
    # 测试一下
    # trainer = Train_all(0, 0, "BJTaxi")  # 这个输入的参数代表训练第几组参数的实验 [k,n]
    # trainer.train_all()

    print(f"step_1={step_1}")

    for step_2 in range(4):
        print(f"--------------No_{step_2} task.-----------------")

        # 初始化训练器 dataset BJTaxi NYCBike1 NYCBike2 NYCTaxi
        trainer = UMSST(step_1, step_2, "NYCTaxi")  # 这个输入的参数代表训练第几组参数的实验 [k,n]
        trainer.train_all()
        # result_mean, result_std = trainer.get_averange()
        # print(f"均值为{result_mean},方差为{result_std}") #数据含义为[[in_MAE,in_MAPE],[out_MAE,out_MAPE]]
        # trainer.save_data()
        print(f"-------------task{step_2} ending--------------")
        print()
    pass


def main_one_pretrain(dataset):

    trainer = UMSST(1, 1, dataset)
    trainer.train_one_pre_train()





if __name__ == '__main__':
    # 测试一下
    trainer = UMSST()  # 这个输入的参数代表训练第几组参数的实验 [k,n]
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='STSSL', type=str, help='training model')
    parser.add_argument('--config_filename', default=f'configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    # parser.add_argument('--dataset',default=f'NYCBike1', help='dataset')
    parser.add_argument('--step_1', default='0', type=int, help='the group of k,from 0 to 3, each means [1, 2, 3, 4] ')
    parser.add_argument('--step_2', default='0', type=int,
                        help='the group of n,from 0 to 3,each means [1, 1.5, 2, 2.5]')
    parser.add_argument('--set_attenuation', default=True, type=eval,
                        help='whether to use the attenuation function,False or True')
    parser.add_argument('-mode',default='pretrain',type=str,help='training mode')

    args1 = parser.parse_args()

    print(f'Starting experiment with configurations in {args1.config_filename}...')

    # print(args1.set_attenuation)
    trainer.step_1 = args1.step_1
    trainer.step_2 = args1.step_2
    # self.set_attenuation = args1.set_attenuation
    time.sleep(1)

    configs = yaml.load(
        open(args1.config_filename),
        Loader=yaml.FullLoader
    )

    # print(args1.set_attenuation)
    configs['set_attenuation'] = args1.set_attenuation

    if args1.mode == 'pre_train':
        configs['train_mode'] = 'pre_train'
        args = argparse.Namespace(**configs)
        # print(args.num_nodes)
        args.model = args1.model

        trainer.train_all(args)

    elif args1.mode == 'base':
        configs['train_mode'] = 'base'
        args = argparse.Namespace(**configs)
        # print(args.num_nodes)
        args.model = args1.model
        trainer.train_baseline_base(args)

    # trainer.train_baseline_base()

    # main_total(3)

    # main_one_pretrain('BJTaxi')

    # 成功就用来保存


    # BJTaxi剩下部分
    # trainer.train_bjtaxi()
