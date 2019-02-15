import os
import kaldi_io
import itertools
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_min_dcf import ComputeMinDcf
from sklearn.metrics import roc_curve

def cosine_sim(a, b):
    return a @ b.T

def compute_error(scores, preds, labels, verbose=True):
    if isinstance(labels, list):
        labels = np.array(labels)
    
    if not labels.sum() == 0:
        fnr = np.count_nonzero((preds == 0) & (labels == 1)) / np.count_nonzero(labels == 1)
    else:
        fnr = 0
    fpr = np.count_nonzero((preds == 1) & (labels == 0)) / np.count_nonzero(labels == 0)
    err = np.count_nonzero(preds != labels) / len(labels)
    eer = compute_eer(scores[labels==1], scores[labels==0])
    
    return eer, err, fpr, fnr

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fprs, tprs, thres = roc_curve(label_vector, score_vector, pos_label=1)
    fnrs = 1 - tprs
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = np.min([fpr[eer_idx], 1-tpr[eer_idx]])
    eer_fpr = fprs[eer_idx]
    eer_fnr = fnrs[eer_idx]
    minDCF = ComputeMinDcf(fnrs, fprs, thres, 0.01, 1, 10) 
#     print("minDCF: {:.4f}, thesh:{:.4f}".format(minDCF[0], minDCF[1]))

    return eer, fpr, fnr

def get_embeds(ids, sv_embeds, id2idx, norm=True):
    idx = [id2idx[id_] for id_ in ids]
    embeds = sv_embeds[idx]
    if norm:
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds

def compute_plda_score(enr_embeds, test_embeds, plda_dir, score_dir="./plda_score"):
    enr_keys = ['enr{}'.format(i) for i in range(len(enr_embeds))]
    test_keys = ['test_{}'.format(i) for i in range(len(test_embeds))]
    keys = enr_keys + test_keys
    embeds = np.concatenate([enr_embeds, test_embeds])

    # write feat
    if not os.path.isdir(score_dir):
        os.makedirs(score_dir)
    ark_scp_output='ark:| copy-vector ark:- ark,scp:{output}/feats.ark,{output}/feats.scp'.format(output=score_dir)
    with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
        for key, vec in zip(keys, embeds):
            kaldi_io.write_vec_flt(f, vec.squeeze(), key=str(key))

    # write trials
    with open("{}/kaldi_trial".format(score_dir), "w") as f:
        trial_pairs = itertools.product(enr_keys, test_keys)
        for pair in trial_pairs:
            f.write(" ".join(pair))
            f.write("\n")

    # scoring call
    ret = subprocess.call("./plda_score.sh {} {}".format(plda_dir, score_dir), shell=True)
    if ret != 0:
        print("plda scoring fails")
        raise ValueError 
    
    # read plda scores
    plda_scores = pd.read_csv(
        "{}/plda_scores".format(score_dir), delimiter=" ", 
        names=["enroll", "test", "score"]).score
    plda_scores = np.array(plda_scores)
    plda_scores = plda_scores.reshape(len(enr_keys), -1)
    assert plda_scores.shape[1] == len(test_embeds)
    
    return plda_scores 

def run_trial(
        enr_embeds, test_embeds, labels, threshold=0.56, 
        plot=True, title="", verbose=True, 
        neg_embeds=None, max_pred=False, min_pred=False,
        plda_dir=None):
    if not plda_dir:
        score = cosine_sim(enr_embeds, test_embeds)
        score_fusion = score.mean(0)
    #     score_fusion_llr = np.log(score_fusion / (1-score_fusion))
    else:
        score = compute_plda_score(enr_embeds, test_embeds, plda_dir)
        score_fusion = score.mean(0)
    pred = score_fusion > threshold
    
    if neg_embeds is not None:
        neg_score = cosine_sim(neg_embeds, test_embeds)
        neg_score_fusion = neg_score.max(0)
        neg_mask = score_fusion > neg_score_fusion
        pred = pred & neg_mask 
    
    if max_pred:
        score_max = score.max(0)
        max_pred = score_max > 0.8
        pred = pred | max_pred
    
    if min_pred:
        score_min = score.min(0)
        min_pred = score_min > 0.5
        pred = pred & min_pred
    
    eer, err, fpr, fnr = compute_error(score_fusion, pred, labels, verbose=True)
    if verbose:
        print("eer={:.4f}, err={:.4f}, fpr={:.4f}, fnr={:.4f}".format(eer, err, fpr, fnr))
    if plot: 
        plot_score(score_fusion, labels, threshold, title)
    
   
    return score

def plot_score(scores, labels, threshold, title):
    plt.figure(figsize=(20,5))
#     plt.ylim([0, 1])
    plt.title(title, fontsize=20)

    pos_score_idx = np.nonzero(labels)[0]
    neg_score_idx = np.nonzero(1-np.array(labels))[0]
    
#     from sklearn.preprocessing import MinMaxScaler
#     min_max_scaler = MinMaxScaler()
#     scores = min_max_scaler.fit_transform(scores.reshape(-1, 1))
#     threshold = min_max_scaler.transform(threshold)[0,0]
    
    plt.plot([threshold]*len(scores), color='k')
    plt.scatter(pos_score_idx, scores[pos_score_idx], alpha=0.5, color='r')
    plt.scatter(neg_score_idx, scores[neg_score_idx], alpha=0.5, color='b')
    plt.show()