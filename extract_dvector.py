# coding: utf-8
import os
import pickle

from sv_system.data.dataloader import init_default_loader
from sv_system.utils.parser import score_parser, set_score_config
from sv_system.data.data_utils import find_dataset
from sv_system.eval.score_utils import embeds_utterance
from sv_system.train.train_utils import load_checkpoint, get_dir_path

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)
if config['output_folder'] is None:
    output_folder = get_dir_path(config['input_file'])
else:
    output_folder = config["output_folder"]

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
dfs, datasets = find_dataset(config, split=False)
si_df, sv_df = dfs
si_dset, sv_dset = datasets
model, _ = load_checkpoint(config)

if not config['lda_file']:
    lda = None
else:
    lda = pickle.load(open(config['lda_file'], "rb"))

#########################################
# Compute Train Embeddings
#########################################
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

val_dataloader = init_default_loader(config, si_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)
dvec_dict = dict(zip(si_df.index.tolist(),
    embeddings.numpy()))

if not config['lda_file']:
    pickle.dump(dvec_dict, open(os.path.join(output_folder,
        "train_dvectors.pkl"), "wb"))
else:
    pickle.dump(dvec_dict, open(os.path.join(output_folder,
        "train_dvectors_lda.pkl"), "wb"))

#########################################
# Compute Test Embeddings
#########################################
val_dataloader = init_default_loader(config, sv_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)
dvec_dict = dict(zip(sv_df.index.tolist(),
    embeddings.numpy()))

if not config['lda_file']:
    pickle.dump(dvec_dict, open(os.path.join(output_folder,
        "test_dvectors.pkl"), "wb"))
else:
    pickle.dump(dvec_dict, open(os.path.join(output_folder,
        "test_dvectors_lda.pkl"), "wb"))
