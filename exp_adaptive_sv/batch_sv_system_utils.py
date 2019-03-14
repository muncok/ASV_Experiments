import os
import kaldi_io
import itertools
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_min_dcf import ComputeMinDcf
from sklearn.metrics import roc_curve

def cosine_sim(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    return a @ b.T

def compute_error(preds, labels, verbose=True):
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)

    if not labels.sum() == 0:
        fnr = np.count_nonzero((preds == 0) & (labels == 1)) / np.count_nonzero(labels == 1)
    else:
        fnr = 0
    fpr = np.count_nonzero((preds == 1) & (labels == 0)) / np.count_nonzero(labels == 0)
    err = np.count_nonzero(preds != labels) / len(labels)

    return {'error':err, 'fpr':fpr, 'fnr':fnr}

def compute_eer(scores, labels):
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)

    pos_scores = scores[np.nonzero(labels==1)]
    neg_scores = scores[np.nonzero(labels==0)]
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fprs, tprs, thres = roc_curve(label_vector, score_vector, pos_label=1)
    fnrs = 1 - tprs

    eer_idx = np.nanargmin(np.abs(fprs - fnrs))
    eer = np.mean([fprs[eer_idx], fnrs[eer_idx]])
    eer_fpr = fprs[eer_idx]
    eer_fnr = fnrs[eer_idx]
    eer_thresh = thres[eer_idx]

    return eer, eer_fpr, eer_fnr, eer_thresh

def compute_minDCF(scores, labels, p_tar=0.5, c_miss=1, c_fa=10):
    if isinstance(labels, list):
        labels = np.array(labels)

    pos_scores = scores[labels==1]
    neg_scores = scores[labels==0]
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fprs, tprs, thres = roc_curve(label_vector, score_vector, pos_label=1)
    fnrs = 1 - tprs

    minDCF, minDCF_thres = ComputeMinDcf(fnrs, fprs, thres, p_tar, c_miss, c_fa)

    return minDCF, minDCF_thres

def get_embeds(ids, sv_embeds, id2idx, norm=True):
    idx = [id2idx[id_] for id_ in ids]
    embeds = sv_embeds[idx]
    if len(embeds) == 1:
        embeds = embeds.reshape(1, -1)
    if norm:
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds

def compute_plda_score(enr_embeds, test_embeds, plda_dir, all_pair=True):
    os.environ["KALDI_ROOT"] = "/host/projects/kaldi"
    enr_keys = ['enr{}'.format(i) for i in range(len(enr_embeds))]
    test_keys = ['test_{}'.format(i) for i in range(len(test_embeds))]
    keys = enr_keys + test_keys
    embeds = np.concatenate([enr_embeds, test_embeds])


    # write trials
    score_dir = TemporaryDirectory()
    with open("{}/kaldi_trial".format(score_dir.name), "w") as f:
        if all_pair:
            trial_pairs = itertools.product(enr_keys, test_keys)
        else:
            trial_pairs = zip(enr_keys, test_keys)
        for pair in trial_pairs:
            f.write(" ".join(pair))
            f.write("\n")

    # write feat
    ark_scp_output='ark:| copy-vector ark:- ' +\
    'ark,scp:{output}/feats.ark,{output}/feats.scp'.format(output=score_dir.name)
    with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
        for key, vec in zip(keys, embeds):
            kaldi_io.write_vec_flt(f, vec.squeeze(), key=str(key))

    # call scoring
    ret = subprocess.call(["./plda_score.sh", plda_dir, score_dir.name])
    if ret != 0:
        print("plda scoring fails")
        raise ValueError

    # read plda scores
    plda_scores = pd.read_csv(
        "{}/plda_scores".format(score_dir.name), delimiter=" ",
        names=["enroll", "test", "score"]).score
    plda_scores = np.array(plda_scores)
    plda_scores = plda_scores.reshape(len(enr_keys), -1)
    if all_pair:
        assert plda_scores.shape[1] == len(test_embeds)

    score_dir.cleanup()

    return plda_scores

def compute_lda_score(enr_embeds, test_embeds, lda_dir, score_dir="./lda_score"):
    os.environ["KALDI_ROOT"] = "/host/projects/kaldi"
    enr_keys = ['enr{}'.format(i) for i in range(len(enr_embeds))]
    test_keys = ['test_{}'.format(i) for i in range(len(test_embeds))]
    keys = enr_keys + test_keys
    embeds = np.concatenate([enr_embeds, test_embeds])

    if not os.path.isdir(score_dir):
        os.makedirs(score_dir)

    # write trials
    with open("{}/kaldi_trial".format(score_dir), "w") as f:
        trial_pairs = itertools.product(enr_keys, test_keys)
        for pair in trial_pairs:
            f.write(" ".join(pair))
            f.write("\n")

    # write feat
    ark_scp_output='ark:| copy-vector ark:- ark,scp:{output}/feats.ark,{output}/feats.scp'.format(output=score_dir)
    with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
        for key, vec in zip(keys, embeds):
            kaldi_io.write_vec_flt(f, vec.squeeze(), key=str(key))

    # scoring call
    ret = subprocess.call("./lda_score.sh {} {}".format(lda_dir, score_dir), shell=True)
    if ret != 0:
        print("lda scoring fails")
        raise ValueError

    # read lda scores
    lda_scores = pd.read_csv(
        "{}/lda_scores".format(score_dir), delimiter=" ",
        names=["enroll", "test", "score"]).score
    lda_scores = np.array(lda_scores)
    lda_scores = lda_scores.reshape(len(enr_keys), -1)
    assert lda_scores.shape[1] == len(test_embeds)

    return lda_scores

def run_trial(
        enr_embeds, test_embeds, labels,
        neg_embeds=None, max_pred=False, plda_dir=None,
        plot=False, title="", verbose=False):

    labels = np.array(labels)

    # scoring
    if not plda_dir:
        score = cosine_sim(enr_embeds, test_embeds)
        score_fusion = score.mean(0)
#         score_fusion_llr = np.log(score_fusion / (1-score_fusion))
    else:
        score = compute_plda_score(enr_embeds, test_embeds, plda_dir)
        score_fusion = score.mean(0)

    ### Equal Error Rate or Classification
    if neg_embeds is not None:
        # neg enrollment
        if len(neg_embeds) == 0:
            print("empty neg_embeds")
        else:
            if not plda_dir:
                neg_score = cosine_sim(neg_embeds, test_embeds)
                neg_score = np.sort(neg_score, 0)
                neg_score_fusion = neg_score[-3:].mean(0)
#                 neg_score_fusion = neg_score.mean(0)
                neg_mask = neg_score_fusion > 0.7
#                 print(np.count_nonzero(neg_mask))
            else:
                neg_score = compute_plda_score(neg_embeds, test_embeds, plda_dir)
                neg_score = np.sort(neg_score, 0)
                neg_score_fusion = neg_score[-3:].mean(0)
                neg_mask = neg_score_fusion > 5
            score_fusion[neg_mask] -= neg_score_fusion[neg_mask]
#             score_fusion = score_fusion - neg_score_fusion
            eer, eer_fpr, eer_fnr, eer_thresh = compute_eer(score_fusion, labels)
    else:
        eer, eer_fpr, eer_fnr, eer_thresh = compute_eer(score_fusion, labels)

#     minDCF, minDCF_thres = compute_minDCF(score_fusion, labels, 0.01, 1, 1)

#     if max_pred:
#         score_max = score.max(0)
#         max_pred = score_max > 0.8
#         pred = pred | max_pred

    if verbose:
        print("[eer] eer={:.4f}, fpr={:.4f}, fnr={:.4f}, thres={:.4f}".format(
            eer, eer_fpr, eer_fnr, eer_thresh))

    if plot:
        if neg_embeds is not None:
            plot_score(neg_score_fusion, labels, eer_thresh, "negative scores")
        plot_score(score_fusion, labels, eer_thresh, title)

    return score_fusion, score

def plot_score(scores, labels, threshold, title):
    plt.figure(figsize=(20,5))
    plt.title(title, fontsize=20)

    pos_score_idx = np.nonzero(labels)[0]
    neg_score_idx = np.nonzero(1-np.array(labels))[0]

#     from sklearn.preprocessing import MinMaxScaler
#     min_max_scaler = MinMaxScaler()
#     scores = min_max_scaler.fit_transform(scores.reshape(-1, 1))
#     threshold = min_max_scaler.transform(threshold)[0,0]
#     plt.ylim([-20, 20])
    plt.plot([threshold]*len(scores), color='k')
    plt.scatter(neg_score_idx, scores[neg_score_idx], alpha=0.2, color='b')
    plt.scatter(pos_score_idx, scores[pos_score_idx], alpha=0.5, color='r')
    plt.show()
