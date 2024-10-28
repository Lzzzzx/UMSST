import argparse
import configparser

def get_predictor_params(args):
    # get the based paras of predictors
    config_file = './configs/params_predictors.conf'
    config = configparser.ConfigParser()
    config.read(config_file)
    # print(config.items())
    # print(config['train']['batch_size'])

    parser_pred = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')
    # train
    parser_pred.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser_pred.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser_pred.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser_pred.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser_pred.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser_pred.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser_pred.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser_pred.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser_pred.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser_pred.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser_pred.add_argument('--debug', default=config['train']['debug'], type=eval)
    parser_pred.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

    if args.model == 'STGCN':
        from model.STGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred,args)
    elif args.model == 'STSGCN':
        from model.STSGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred,args)
    elif args.model == 'STFGNN':
        from model.STFGNN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred,args)
    elif args.model == 'GMAN':
        from model.GMAN.args import parse_args
        args_predictor=parse_args(args.dataset, parser_pred,args)
    # elif args.model=='STSSL':
    #     from model.ST_SSL.args import parse_args
    #     args_predictor = parse_args(args.dataset, parser_pred)
    else:
        raise ValueError

    return args_predictor