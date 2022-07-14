import os.path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from opacus import PrivacyEngine
import numpy as np
import copy
import wandb


from data import getDL
from nn import NoisyNN, SGLDOptim
from train import train_model
from utils import acc_score_fn, _saveMeta, getID, _saveParams

def newTrainedModel(cfg, model_id):
        tag = cfg['tag']
        save_model = cfg['save_model']
        save_model_path = cfg['save_model_path']
        clip_val = cfg['clip_val']
        cuda_device_id = cfg['cuda_device_id']

        print("Creating victim with tag = " + str(tag))
        model = NoisyNN(cfg)
        if cuda_device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cuda_device_id)
                                  if torch.cuda.is_available() else "cpu")

        model_ft = model.nn
        model_ft.to(device)

        use_batch_sampler = cfg['opacus']
        t_dl = getDL(cfg, train = True, batch_size = cfg['batch_size'],
                     adversial = tag, use_batch_sampling = use_batch_sampler)
        v_dl = getDL(cfg, train = False, batch_size = cfg['batch_size'],
                     adversial = tag, use_batch_sampling = False)

        ds_size = t_dl.ds_size
        if not cfg['opacus']:
            lr = cfg['lr']
            criterion = nn.CrossEntropyLoss(reduction = "sum")
            optimizer = SGLDOptim(model_ft.parameters(), lr, weight_decay = 1,
                                  data_size = ds_size, cfg = cfg)
        else:
            assert clip_val > 0
            lr = cfg['lr']*ds_size/2
            sigma = 2/ds_size*1/np.sqrt(cfg['lr'])/clip_val
            weight_decay = 1/ds_size

            criterion = nn.CrossEntropyLoss(reduction = "mean")
            optimizer = SGD(model_ft.parameters(), lr = lr, momentum = 0,
                            weight_decay = weight_decay)
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            privacy_engine = PrivacyEngine(
                model_ft,
                sample_rate = t_dl.sample_rate,
                alphas = alphas,
                noise_multiplier = sigma,
                max_grad_norm = clip_val,
                secure_rng = False,
            )
            privacy_engine.attach(optimizer)

        if cfg['scheduler_type'] == 'StepLR':
            ms = cfg['sched_milestones']
            gamma = cfg['sched_gamma']
            scheduler = MultiStepLR(optimizer, milestones = ms, gamma = gamma)
        elif cfg['scheduler_type'] == 'Cosine':
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=cfg['epochs'])
        else:
            scheduler = None

        score_fn = acc_score_fn
        dataloaders = {'train' : t_dl, 'val': v_dl}
        meta = train_model(cfg, model, criterion, optimizer, dataloaders,
                           score_fn, scheduler, device)

        meta['batch_size'] = cfg['batch_size']
        meta['lr'] = lr,
        meta['epochs'] = cfg['epochs']
        meta['model_id'] = model_id
        meta['nn_type'] = cfg['nn_type']
        meta['tag'] = tag

        _saveMeta(save_model_path, model_id, meta)
        if save_model:
            model.saveWeights(save_model_path + model_id)

        return model

def ntmWandbWrapper(cfg):
    wandb_config = copy.deepcopy(cfg)
    model_id = getID(cfg['tag'])
    wandb_config['model_id'] = model_id

    with wandb.init(name=cfg['wandb_run_name'],
                    project = cfg['wandb_project'],
                    tags = [], entity = 'hellerguyh',
                    config = wandb_config) as wandb_run:
        model = newTrainedModel(cfg, model_id)
        _saveParams(cfg['save_model_path'], model_id, cfg)


def getParser():
    import argparse
    parser = argparse.ArgumentParser(description = "Train a model")
    parser.add_argument("--tag", action = "store_true")
    parser.add_argument("--pred_mal_samples", action = "store_true")
    parser.add_argument("--nn_type", choices = ['LeNet5'])
    parser.add_argument("--ds_name", choices = ['MNIST'])
    parser.add_argument("--save_model_path", type = str)
    parser.add_argument("--cuda_device_id", type = int)
    parser.add_argument("--repeat", type = int, default = 1)
    parser.add_argument("--epochs", type = int)
    parser.add_argument("--lr", type = float, help = "learning rate")
    parser.add_argument("--batch_size", type = int, help = "batch size")
    parser.add_argument("--clip_val", type = float,
                        help = "Clip gradients to this value")
    parser.add_argument("--scheduler_type", type=str,
                        choices = ['StepLR', 'Cosine', 'None'])
    parser.add_argument("--sched_milestones", nargs="+", type=float)
    parser.add_argument("--sched_gamma", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--normalize", type = bool,
                        help = "use normalization augmentation")
    parser.add_argument("--adv_sample_img", type = str, help = "path to"\
                        "adversial sample", default = None)
    parser.add_argument("--wandb_run_name", type = str, required=True)
    parser.add_argument("--wandb_project", type = str, required=True)
    return parser

def main():
    from lenet_mnist_default_cfg import cfg
    parser = getParser()
    args = parser.parse_args()
    if args.save_model_path is not None:
        args.save_model_path = os.path.normpath(args.save_model_path) + "/"

    for arg in args.__dict__.keys():
        if args.__dict__[arg] is not None:
            cfg.update({arg : args.__dict__[arg]})
    cfg.update({'adv_sample_img' : args.adv_sample_img})
    if cfg['clip_val'] > 0:
        cfg['opacus'] = True
    for i in range(args.repeat):
        print("Training model %d"%i)
        cfg_i = copy.deepcopy(cfg)
        ntmWandbWrapper(cfg_i)


if __name__ == "__main__":
    main()
