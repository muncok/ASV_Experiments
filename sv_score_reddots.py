# coding: utf-8
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from sv_system.data.dataloader import init_default_loader
from sv_system.model import find_model
from sv_system.utils.parser import default_config, score_parser, set_input_config, set_train_config
from sv_system.data.dataset import SpeechDataset
from sv_system.eval.score_utils import embeds_utterance

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
model = args.model
dataset = args.dataset

si_config = default_config(model)
si_config = set_input_config(si_config, args)
si_config = set_train_config(si_config, args)

#########################################
# Model Initialization
#########################################
si_config["input_dim"] = 40
model, criterion = find_model(si_config, model, 1)
# model.load("models/compare_train_methods/reddots/si_reddots_TdnnModel_3s_0.1s_fbank_full_frame.pt")
lda = None

#########################################
# Load trial
#########################################
ndx = pd.read_pickle("dataset/dataframes/reddots//m_part4_tp/m_part4_tp_ndx.pkl")
trn = pd.read_pickle("dataset/dataframes/reddots//m_part4_tp/m_part4_tp_trn.pkl")
cord = pickle.load(open("dataset/dataframes/reddots/m_part4_tp/ndx_idxs.pkl", "rb"))

err_type = {0:'TC', 1:'TW', 2:'IC', 3:'IW'}

# Enrollment (trn)
si_config['data_folder'] = "/home/muncok/DL/dataset/SV_sets/reddots_r2015q4_v1/wav/"
trn_dataset = SpeechDataset.read_df(si_config, trn, "test")
val_dataloader = init_default_loader(si_config, trn_dataset, shuffle=False)
trn_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)
trn_id = list(trn.id.unique())
spk_model_dict = {}
for id in trn_id:
    index = np.nonzero(trn.id == id)
    spk_model_dict[id] = trn_embeddings[index].mean(0, True)
spk_models = torch.cat([emb for emb in spk_model_dict.values()])

# SV Scoring (ndx)
ndx_file = pd.DataFrame(ndx.file.unique().tolist(), columns=['file'])
ndx_dataset = SpeechDataset.read_df(si_config, ndx_file, "test")
val_dataloader = init_default_loader(si_config, ndx_dataset, shuffle=False)
ndx_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)

sim_matrix = F.cosine_similarity(spk_models.unsqueeze(1), ndx_embeddings.unsqueeze(0), dim=2)
sims = sim_matrix[cord]

scores = dict()
for t in range(4):
    trial_type_idx = ndx[ndx.trial_type == t].index
    scores[t] = sims[trial_type_idx]

for t in range(4):
     print("{} mean:{:.2f}, std:{:.3f}".format(err_type[t], scores[t].mean(), scores[t].std()))

# TD EERs
for t in range(1,4):
    score_vector = np.concatenate((scores[0], scores[t]))
    label_vector = np.concatenate((np.ones(len(scores[0])),
                               np.zeros(len(scores[t]))))
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    print("[{}] eer: {:.2f}, thres: {:.5f}".format(err_type[t], eer, thres))


# TI EERs
score_vector = np.concatenate((scores[0], scores[1],
                              scores[2], scores[3]))
label_vector = np.concatenate((np.ones(len(scores[0]) + len(scores[1])),
                           np.zeros(len(scores[2]) + len(scores[3]))))
fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
print("[TI] eer: {:.2f}".format(eer))

