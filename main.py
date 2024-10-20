import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
sys.path.append('..')
import yaml 
import argparse
import traceback
import time
import torch

from model.models import STSSL
from model.trainer import Trainer
from lib.dataloader import Get_Dataloader_STssl
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, 
)

def  model_supervisor(args):
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
    
    ## init model and set optimizer
    model = STSSL(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters, 
        lr=args.lr_init, 
        eps=1.0e-8, 
        weight_decay=0, 
        amsgrad=False
    )

    ## start training
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        dataloader=dataloader,
        graph=graph, 
        args=args
    )
    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
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

if __name__=='__main__':

    dataset='NYCBike1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=f'configs/{dataset}.yaml',
                    type=str, help='the configuration to use')
    args = parser.parse_args()
    
    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    results=[]
    # seed = [36,10,15,31, 42, 53, 64, 75,99,87,123] # 用更多种子跑
    seed=[10,15,31]
    for i in range(len(seed)):
        args.seed=seed[i]
        result=model_supervisor(args)
        results.append(result['test_results'])

    results_mean = np.mean(results, axis=0)
    results_std = np.std(results, axis=0)
    save_data = pd.DataFrame({'in_MAE_mean': [results_mean[0][0]],
                              'out_MAE_mean': [results_mean[1][0]],
                              'in_MAPE_mean': [results_mean[0][1]],
                              'out_MAPE_mean': [results_mean[1][1]],
                              'in_MAE_std': [results_std[0][0]],
                              'out_MAE_std': [results_std[1][0]],
                              'in_MAPE_std': [results_std[0][1]],
                              'out_MAPE_std': [results_std[1][1]],
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
    save_data_every = pd.DataFrame({'seed': seed,
                                    'in_MAE': in_mae,
                                    'out_MAE': out_mae,
                                    'in_MAPE': in_mape,
                                    'out_MAPE': out_mape
                                    }).set_index('seed')

    # 保存均值方差
    # save_data.to_csv(f"res/{dataset}_ST_SSL.csv")
    # print(f'已经保存至 res/{dataset}_ST_SSL.csv文件中')

    # 保存每阶段结果
    save_data_every.to_csv(f"res/{dataset}_every.csv")
    print(f'已经保存至 res/{dataset}_every.csv文件中')