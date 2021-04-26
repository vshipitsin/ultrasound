import os
import yaml


def get_params():
    with open(os.path.join("configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)

    params = {}
    params.update({'data_path': configs['paths']['data'],
                   'log_dir': configs['paths']['log_dir']})

    params.update({'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size'])})

    params.update({'adaptive_layer': configs['model_parameters']['AdaptiveLayer']['adjustment']})
    if params['adaptive_layer'] == 'None':
        params['adaptive_layer'] = None
    
    params.update({'init_features': int(configs['model_parameters']['UNet']['init_features']),
                   'depth': int(configs['model_parameters']['UNet']['depth'])})

    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})

    return params
