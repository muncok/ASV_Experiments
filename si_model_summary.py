# coding: utf-8
import argparse
import numpy as np
# from train.train_utils import  load_checkpoint
from sv_system.model.model_utils import find_model
# from data.dataloader import init_default_loader
# from train.si_train import val


#########################################
# Parser
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('-arch', type=str, help='model architecture')
args = parser.parse_args()

#########################################
# Model parameters
#########################################
config = dict(arch=args.arch, n_labels=1, gpu_no=[1], no_cuda=True)
model = find_model(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("{}'s model parameters: {}".format(args.arch, params))
