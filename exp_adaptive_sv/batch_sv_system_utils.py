from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt

def cosine_sim(a, b):
    return a @ b.T

def compute_error(preds, labels):
    if isinstance(labels, list):
        labels = np.array(labels)
    
    if not labels.sum() == 0:
        fnr = np.count_nonzero((preds == 0) & (labels == 1)) / np.count_nonzero(labels == 1)
    else:
        fnr = 0
    fpr = np.count_nonzero((preds == 1) & (labels == 0)) / np.count_nonzero(labels == 0)
    err = np.count_nonzero(preds != labels) / len(labels)
    
    return err, fpr, fnr

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return eer

def get_embeds(ids, sv_embeds, id2idx, norm=True):
    idx = [id2idx[id_] for id_ in ids]
    embeds = sv_embeds[idx]
    if norm:
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds

def run_trial(enr_embeds, trial_embeds, labels, threshold=0.56, neg_embeds=None):
    score = cosine_sim(enr_embeds, trial_embeds)
    score_fusion = score.mean(0)
    pred = score_fusion > threshold
    if neg_embeds is not None:
#         ipdb.set_trace()
        neg_score = cosine_sim(neg_embeds, trial_embeds)
        neg_score_fusion = neg_score.max(0)
#         neg_score_fusion = neg_score.mean(0)
        neg_mask = score_fusion > neg_score_fusion
        pred = pred & neg_mask 
    
    err, fpr, fnr = compute_error(pred, labels)
    print("err={:.4f}, fpr={:.4f}, fnr={:.4f}, thres={:.4f}".format(err, fpr, fnr, threshold))
    
    return score_fusion

def plot_score(scores, labels, threshold, title):
    plt.figure(figsize=(20,5))
#     plt.ylim([0, 1])
    plt.title(title, fontsize=20)

    pos_score_idx = np.nonzero(labels)[0]
    neg_score_idx = np.nonzero(1-np.array(labels))[0]
    
    pred = scores > threshold
    err, fpr, fnr = compute_error(pred, labels)
    try:
        eer = compute_eer(scores[pos_score_idx], scores[neg_score_idx])
    except:
        eer = 0
    print("err={:.4f}, eer={:.4f}, fpr={:.4f}, fnr={:.4f}, thres={:.4f}".format(err, eer, fpr, fnr, threshold))

    plt.plot([threshold]*len(scores), color='k')
    plt.scatter(pos_score_idx, scores[pos_score_idx], alpha=0.5, color='r')
    plt.scatter(neg_score_idx, scores[neg_score_idx], alpha=0.5, color='b')
    plt.show()