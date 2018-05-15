# coding: utf-8
import os

import pandas as pd
import argparse

from sv_system.data import dataloader as dloader
from sv_system.data.dataset import SpeechDataset
from sv_system.model.TDNN import TdnnModel
from sv_system.train import si_train
from sv_system.utils import secToSample, secToFrames
from sv_system.utils.data_utils import split_df
from sv_system.utils.parser import test_config

#########################################
# Parser
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('-nep', '--epochs',
                    type=int,
                    help='number of epochs to train for',
                    default=140)

parser.add_argument('-dataset',
                    type=str,
                    help='dataset',
                    default='voxc')

parser.add_argument('-inLen', '--input_length',
                    type=int,
                    help='length of input audio, sec',
                    default=3)

parser.add_argument('-spLen', '--splice_length',
                    type=float,
                    help='length of spliced audio snippet, sec',
                    default=0.2)

#########################################
# Configuration
#########################################
args = parser.parse_args()
model = "TDNN"
dataset = args.dataset
si_config = test_config(model)

si_config['input_clip'] = True
sec_input = args.input_length
sec_splice = args.splice_length
si_config['input_length'] = secToSample(sec_input)
si_config['input_frames'] = secToFrames(sec_input)
si_config['splice_frames'] = secToFrames(sec_splice)
si_config['input_format'] = 'fbank'

si_train.set_seed(si_config)
si_config['n_epochs'] = args.epochs
si_config['print_step'] = 100
si_config['lr'] = [0.001, 0.0001]
si_config['schedule'] = [20]

if dataset == "voxc":
    si_config['data_folder'] = "../dataset/voxceleb/"
    df = pd.read_pickle("../dataset/dataframes/Voxc_Dataframe.pkl")
    n_labels = 1260
elif dataset == "reddots":
    si_config['data_folder'] = "../dataset/reddots_r2015q4_v1/wav"
    df = pd.read_pickle("../dataset/dataframes/Reddots/Reddots_Dataframe.pkl")
    n_labels = 70
else:
    raise FileNotFoundError

si_config['output_file'] = "../models/compare_train_methods/" \
                           "{dset}/si_{dset}_{model}_{in_len}s_" \
                           "{s_len}s_{in_format}.pt".format(dset=dataset, in_len=sec_input,
                                                            s_len=sec_splice,
                                                            in_format=si_config['input_format'],
                                                            model=model)

#########################################
# Model Initialization
#########################################
si_model = TdnnModel(si_config, n_labels)
if os.path.isfile(si_config['output_file']):
    si_model.load(si_config['output_file'])

#########################################
# Model Training
#########################################

split_dfs = split_df(df)
loaders = []
for i, df in enumerate(split_dfs):
    if i == 0:
        dataset = SpeechDataset.read_df(si_config, df, "train")
        loader = dloader.init_default_loader(si_config, dataset, True)
    else:
        dataset = SpeechDataset.read_df(si_config, df, "test")
        loader = dloader.init_default_loader(si_config, dataset, False)
    loaders.append(loader)

si_train.si_tdnn_train(si_config, model=si_model, loaders=loaders)

#########################################
# Model Evaluation
#########################################
si_train.evaluate(si_config, si_model, loaders[-1])
