# coding: utf-8
import os
import numpy as np
import pandas as pd

from sv_system.utils.parser import test_config
from sv_system.model.SpeechModel import SpeechModel, SpeechResModel
from sv_system.model.AuxModels import SimpleCNN, LongCNN
from sv_system.data.dataset import SpeechDataset
from sv_system.utils import secToSample, secToFrames
from sv_system.utils.data_utils import split_df

import sv_system.train.si_train as si_train
import sv_system.data.dataloader as dloader



#########################################
# Configuration
#########################################
model = "LongCNN"
si_config = test_config(model)
si_config["num_workers"] = 32
si_config["batch_size"] = 128
#########################################
# Input configure
#########################################
si_config['input_clip'] = True
si_config['input_length'] = secToSample(3)
si_config['splice_frames'] = secToFrames(0.1)
si_config['input_format'] = 'fbank'

#########################################
# Model Initialization
#########################################
si_model = SimpleCNN(si_config, 1260)
# si_model.load("models/voxc/si_train/full_train/si_voxc_longcnn_3s_3s_full.pt")

#########################################
# Model Training
#########################################
si_train.set_seed(si_config)
si_config['n_epochs'] = 50
si_config['print_step'] = 100
si_config['output_file'] = "models/voxc/si_train/full_train/si_voxc_simplecnn_3s_0.1s_full.pt"
si_config['data_folder'] = "dataset/voxceleb/"

voxc_df = pd.read_pickle("dataset/dataframes/Voxc_Dataframe.pkl")
split_dfs = split_df(voxc_df)
loaders = []
for i, df in enumerate(split_dfs):
    if i == 0:
        dataset = SpeechDataset.read_df(si_config, df, "train")
        loader = dloader.init_default_loader(si_config, dataset, True)
    else:
        dataset = SpeechDataset.read_df(si_config, df, "test")
        loader = dloader.init_default_loader(si_config, dataset, False)
    loaders.append(loader)

si_train.si_train(si_config, model=si_model, loaders=loaders)
# si_train.si_agg_train(si_config, model=si_model, loaders=loaders)

#########################################
# Model Evaluation
#########################################
# si_train.agg_evaluate(si_config, si_model, loaders[-1])
