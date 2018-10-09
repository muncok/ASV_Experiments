# coding: utf-8
import os
import pickle
import numpy as np

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

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

#########################################
# Compute Train Embeddings
#########################################

si_dataloader = init_default_loader(config, si_dset, shuffle=False)
si_embeddings, _ = embeds_utterance(config, si_dataloader, model, lda)
si_keys = si_df.index.tolist()
pickle.dump(si_keys, open(os.path.join(output_folder, "si_keys.pkl"), "wb"))
np.save(os.path.join(output_folder, "si_embeds.npy"), si_embeddings)

si_feat = dict(zip(si_keys, si_embeddings.numpy()))
pickle.dump(si_feat, open(os.path.join(output_folder, "si_feat.pkl"), "wb"))

#########################################
# Compute Test Embeddings
#########################################
sv_dataloader = init_default_loader(config, sv_dset, shuffle=False)
sv_embeddings, _ = embeds_utterance(config, sv_dataloader, model, lda)

sv_keys = sv_df.index.tolist()
sv_embeds = sv_embeddings
pickle.dump(sv_keys, open(os.path.join(output_folder, "sv_keys.pkl"), "wb"))
np.save(os.path.join(output_folder, "sv_embeds.npy"), sv_embeddings)

# sv_feat = dict(zip(sv_keys, sv_embeddings.numpy()))
# pickle.dump(sv_feat, open(os.path.join(output_folder, "sv_feat.pkl"), "wb"))
