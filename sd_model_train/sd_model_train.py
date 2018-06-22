# coding: utf-8
import os

from sv_system.data.dataloader import init_loaders_from_df
from sv_system.data.dataset import find_dataset
from sv_system.model import find_model
from sv_system.train import si_train
from sv_system.utils.data_utils import split_df
from sv_system.utils.parser import default_config, train_parser, set_input_config, set_train_config

#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
model = args.model
dataset = args.dataset

si_config = default_config(model)
si_config = set_input_config(si_config, args)
si_config = set_train_config(si_config, args)
si_config['output_file'] = "models/compare_train_methods/" \
        "{dset}/si_{dset}_{model}_{in_len}f_" \
        "{s_len}f_{in_format}_{suffix}.pt".format(
                dset=dataset, in_len=si_config["input_frames"],
                s_len=si_config["splice_frames"],
                in_format=si_config['input_format'],
                model=model, suffix=args.suffix)

#########################################
# Dataset loaders
#########################################
df, n_labels = find_dataset(si_config, dataset)
split_dfs = split_df(df)
loaders = init_loaders_from_df(si_config, split_dfs)

#########################################
# Model Initialization
#########################################
si_model, criterion = find_model(si_config, model, n_labels)
print(si_model)
# model initialization
if si_config['input_file']:
    si_model.load_partial(si_config['input_file'])
elif args.start_epoch > 0 and os.path.isfile(si_config['output_file']):
    si_model.load(si_config['output_file'])
    print("training start from {} epoch".format(args.start_epoch))

#########################################
# Model Training
#########################################
si_config['print_step'] = 10
si_train.set_seed(si_config)
si_train.si_train(si_config, model=si_model, loaders=loaders,
        criterion=criterion)

#########################################
# Model Evaluation
#########################################
# si_train.evaluate(si_config, si_model, loaders[-1])
