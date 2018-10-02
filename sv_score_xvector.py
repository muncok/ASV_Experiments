# coding: utf-8
import os, pickle
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve

from sv_system.data.data_utils import find_trial
from sv_system.utils.parser import score_parser, set_score_config

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

embeddings = np.load(config['input_file'])
embeddings = torch.from_numpy(embeddings)

#########################################
# Load trial
#########################################
trial = find_trial(config)
sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
score_vector = sim_matrix[cord].numpy()
label_vector = trial.label
fpr, tpr, thres = roc_curve(
        label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

output_folder = config['output_folder']
pickle.dump(score_vector, open(os.path.join(output_folder,
    "cosine_scores_{}.pkl".format(config['score_mode'])), "wb"))
print("[TI] eer: {:.3f}%".format(eer*100))
