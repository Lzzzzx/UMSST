import numpy as np
import configparser
# from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import pandas as pd
import torch


def load_graph(adj_file, device='cpu'):
    '''Loading graph in form of edge index.'''
    graph = np.load(adj_file)['adj_mx']
    graph = torch.tensor(graph, device=device, dtype=torch.float)

    return graph

def parse_args(DATASET, parser, args):
    # get configuration
    if args.train_mode == 'base':
        config_file = './configs/STSGCN/{}_base.conf'.format(DATASET)
    elif args.train_mode == 'pre_train':
        config_file = './configs/STSGCN/{}.conf'.format(DATASET)
    else:
        raise ValueError
    config = configparser.ConfigParser()
    config.read(config_file)

    filter_list_str = config.get('model', 'filter_list')
    filter_list = eval(filter_list_str)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--graph_file', type=str, default=config['data']['graph_file'])

    # model
    parser.add_argument('--filter_list', type=list, default=config['model']['filter_list'])
    parser.add_argument('--rho', type=int, default=config['model']['rho'])
    parser.add_argument('--feature_dim', type=int, default=config['model']['feature_dim'])
    parser.add_argument('--module_type', type=str, default=config['model']['module_type'])
    parser.add_argument('--activation', type=str, default=config['model']['activation'])
    parser.add_argument('--temporal_emb', type=eval, default=config['model']['temporal_emb'])
    parser.add_argument('--spatial_emb', type=eval, default=config['model']['spatial_emb'])
    parser.add_argument('--use_mask', type=eval, default=config['model']['use_mask'])
    parser.add_argument('--steps', type=int, default=config['model']['steps'])
    parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])

    args, _ = parser.parse_known_args()
    # args.filepath = '../data/' + DATASET +'/'
    # args.filename = DATASET

    # args.filepath = '../data/' + DATASET +'/'
    # args.filename = DATASET
    # data = load_st_dataset(DATASET, args_base)

    filename = DATASET

    adj = load_graph(args.graph_file)
    args.filter_list = filter_list
    args.adj = adj
    return args