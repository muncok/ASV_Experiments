# coding: utf-8
import pandas as pd

from ..sv_system.utils.parser import test_config
from ..sv_system.model.TDNN import TdnnModel
from ..sv_system.data.dataset import SpeechDataset
from ..sv_system.utils import secToSample, secToFrames
from ..sv_system.utils.data_utils import split_df

from ..sv_system.train import si_train
from ..sv_system.data import dataloader as dloader



#########################################
# Configuration
#########################################
model = "Tdnn"
si_config = test_config(model)

#########################################
# Input configure
#########################################
si_config['input_clip'] = True
sec_input = 3
si_config['input_length'] = secToSample(sec_input)
si_config['input_frames'] = secToFrames(sec_input)
si_config['splice_frames'] = secToFrames(0.1)
si_config['input_format'] = 'fbank'

#########################################
# Model Initialization
#########################################
si_model = TdnnModel(si_config, 1260)
# si_config['output_file'] = "models/reddots/si_train/si_reddots_tdnnfc_3s_0.1s_mean.pt"
si_config['output_file'] = "models/voxc/si_train/si_voxc_tdnnfc_3s_0.1s_mean.pt"
# si_model.load(si_config['output_file'])
#########################################
# Model Training
#########################################
si_train.set_seed(si_config)
si_config['n_epochs'] = 100
si_config['print_step'] = 200
si_config['data_folder'] = "dataset/voxceleb/"
# si_config['data_folder'] = "dataset/reddots_r2015q4_v1/wav"

df = pd.read_pickle("dataset/dataframes/Voxc_Dataframe.pkl")
# df = pd.read_pickle("dataset/dataframes/Reddots/Reddots_Dataframe.pkl")
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
