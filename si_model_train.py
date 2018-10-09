# coding: utf-8
import sys
import os
import uuid

import torch
from tensorboardX import SummaryWriter

from sv_system.utils.parser import train_parser, set_train_config

from sv_system.data.dataloader import init_loaders
from sv_system.data.data_utils import find_dataset, find_trial

from sv_system.model.model_utils import find_model

from sv_system.train.train_utils import (set_seed, find_optimizer, get_dir_path,
        load_checkpoint, save_checkpoint, new_exp_dir)
from sv_system.train.train_utils import Logger
from sv_system.train.si_train import train, val, sv_test
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR



#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
dataset = args.dataset
config = set_train_config(args)

#########################################
# Dataset loaders
#########################################
_, datasets = find_dataset(config)
loaders = init_loaders(config, datasets)

#########################################
# Model Initialization
#########################################
model= find_model(config)
criterion, optimizer = find_optimizer(config, model)
plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

#########################################
# Model Save Path
#########################################
if config['input_file']:
    # start new experiment continuing from "input_file"
    load_checkpoint(config, model, criterion, optimizer)
    config['output_dir'] = new_exp_dir(config,
            get_dir_path(config['input_file'])[:-4])
else:
    # start new experiment
    config['output_dir'] = new_exp_dir(config)

print("Model will be saved to : {}".format(config['output_dir']))

#########################################
# Logger
#########################################
sys.stdout = Logger(os.path.join(config['output_dir'],
    'log_{}'.format(str(uuid.uuid4())[:5]) + '.txt'))

# tensorboard
log_dir = config['output_dir']
writer = SummaryWriter(log_dir)

#########################################
# dataloader and scheduler
#########################################
if not config['no_eer']:
    train_loader, val_loader, test_loader, sv_loader = loaders
else:
    train_loader, val_loader, test_loader = loaders


#########################################
# trial
#########################################
trial = find_trial(config)
if not config['no_eer']:
    best_metric = config['best_metric'] if 'best_metric' in config \
            else 1.0
else:
    best_metric = config['best_metric'] if 'best_metric' in config \
            else 0.0

#########################################
# Model Training
#########################################

set_seed(config)
for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
    # idx = 0
    # while(epoch_idx >= config['lr_schedule'][idx]):
    # # use new lr from schedule epoch not a next epoch
        # idx += 1
        # if idx == len(config['lr_schedule']):
            # break
    # curr_lr = config['lrs'][idx]
    # optimizer.state_dict()['param_groups'][0]['lr'] = curr_lr
    print("curr_lr: {}".format(curr_lr))

    # train code
    train_loss, train_acc = train(config, train_loader, model, optimizer, criterion)
    writer.add_scalar("train/lr", curr_lr, epoch_idx)
    writer.add_scalar('train/loss', train_loss, epoch_idx)
    writer.add_scalar('train/acc', train_acc, epoch_idx)

    # validation code
    val_loss, val_acc = val(config, val_loader, model, criterion)
    writer.add_scalar('val/loss', val_loss, epoch_idx)
    writer.add_scalar('val/acc', val_acc, epoch_idx)
    print("epoch #{}, val accuracy: {}".format(epoch_idx, val_acc))

    plateau_scheduler.step(train_loss)

    # evaluate best_metric
    if not config['no_eer']:
        # eer validation code
        eer, label, score = sv_test(config, sv_loader, model, trial)
        writer.add_scalar('sv_eer', eer, epoch_idx)
        writer.add_pr_curve('DET', label, score, epoch_idx)
        print("epoch #{}, sv eer: {}".format(epoch_idx, eer))
        if eer < best_metric:
            best_metric = eer
            is_best = True
        else:
            is_best = False
    else:
        if val_acc > best_metric:
            best_metric = val_acc
            is_best = True
        else:
            is_best = False


    # for name, param in model.named_parameters():
        # writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

    filename = config["output_dir"] + \
            "/model.{:.4}.pth.tar".format(curr_lr)

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_checkpoint({
        'epoch': epoch_idx,
        'step_no': (epoch_idx+1) * len(train_loader),
        'arch': config["arch"],
        'dataset': config["dataset"],
        'loss': config["loss"],
        'state_dict': model_state_dict,
        'best_metric': best_metric,
        'optimizer' : optimizer.state_dict(),
        }, epoch_idx, is_best, filename=filename)

#########################################
# Model Evaluation
#########################################
test_loss, test_acc = val(config, test_loader, model, criterion)
