# coding: utf-8
# coding: utf-8
import os
import pandas as pd

from sv_system.data import dataloader as dloader
from sv_system.data.dataset import SpeechDataset
from sv_system.model.TDNN import TdnnModel, TdnnStatModel
from sv_system.model.AuxModels import SimpleCNN
from sv_system.train import si_train
from sv_system.utils import secToSample, secToFrames
from sv_system.utils.data_utils import split_df
from sv_system.utils.parser import test_config, train_parser

#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()

#########################################
# Configuration
#########################################
model = args.model
dataset = args.dataset

sec_input = args.input_length
sec_splice = args.splice_length
si_config = test_config(model)
si_config['input_clip'] = True
si_config['input_length'] = secToSample(sec_input)
si_config['input_frames'] = secToFrames(sec_input)
si_config['splice_frames'] = secToFrames(sec_splice)
si_config['stride_frames'] = args.stride_frames
si_config['input_format'] = 'fbank'

si_config['n_epochs'] = args.epochs
si_config['print_step'] = 100
si_config['lr'] = [0.001, 0.0001]
si_config['schedule'] = [40]
si_config['s_epoch'] = args.start_epoch

si_config['batch_size'] = 1
si_config['num_workers'] = 4

if dataset == "voxc":
    si_config['data_folder'] = "../dataset/voxceleb/"
    df = pd.read_pickle("../dataset/dataframes/Voxc_Dataframe.pkl")
    n_labels = 1260
elif dataset == "reddots":
    si_config['data_folder'] = "../dataset/reddots_r2015q4_v1/wav"
    df = pd.read_pickle("../dataset/dataframes/reddots/Reddots_Dataframe.pkl")
    n_labels = 70
elif dataset == "reddots_vad":
    si_config['data_folder'] = "../vad/reddots_vad/"
    df = pd.read_pickle("/home/muncok/DL/projects/sv_experiments/dataset/dataframes/reddots/reddots_vad.pkl")
    n_labels = 70
else:
    raise FileNotFoundError

si_config['output_file'] = "../models/compare_train_methods/" \
                           "{dset}/si_{dset}_{model}_{in_len}s_" \
                           "{s_len}s_{in_format}_{suffix}.pt".format(dset=dataset, in_len=sec_input,
                                                                     s_len=sec_splice,
                                                                     in_format=si_config['input_format'],
                                                                     model=model, suffix=args.suffix)

#########################################
# Model Initialization
#########################################
if model == "SimpleCNN":
    si_model = SimpleCNN(si_config, n_labels)
elif model == "TdnnModel":
    si_model = TdnnModel(si_config, n_labels)
elif model == "TdnnStatModel":
    si_model = TdnnStatModel(si_config, n_labels)
else:
    print("'{}' is not implemented model".format(model))

if args.start_epoch > 0 and os.path.isfile(si_config['output_file']):
    si_model.load(si_config['output_file'])
    print("training start from {} epoch".format(args.start_epoch))

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

si_train.set_seed(si_config)
if "Tdnn" in model:
    si_train.si_tdnn_train(si_config, model=si_model, loaders=loaders)
elif model == 'SimpleCNN':
    si_train.si_train(si_config, model=si_model, loaders=loaders)

#########################################
# Model Evaluation
#########################################
# si_train.evaluate(si_config, si_model, loaders[-1])
