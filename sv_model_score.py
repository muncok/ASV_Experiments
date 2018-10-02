# coding: utf-8
import os, pickle
import numpy as np

import torch.nn.functional as F

from sklearn.metrics import roc_curve

from sv_system.data.dataloader import init_default_loader
from sv_system.data.data_utils import find_dataset, find_trial
from sv_system.utils.parser import score_parser, set_score_config
from sv_system.eval.score_utils import embeds_utterance
from sv_system.train.train_utils import load_checkpoint, get_dir_path

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
_, datasets = find_dataset(config, split=False)
model, _ = load_checkpoint(config)
lda = None

#########################################
# Compute Embeddings
#########################################
sv_dset = datasets[-1]
val_dataloader = init_default_loader(config, sv_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)

#########################################
# Load trial
#########################################
trial = find_trial(config)

#########################################
# Scoring
#########################################
sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
score_vector = sim_matrix[cord].numpy()
label_vector = trial.label
fpr, tpr, thres = roc_curve(
        label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
print("[TI] eer: {:.3f}%".format(eer*100))

#########################################
# Save score
#########################################
output_folder = get_dir_path(config['input_file'])
pickle.dump(score_vector, open(os.path.join(output_folder,
    "cosine_scores_{}.pkl".format(config['score_mode'])), "wb"))

