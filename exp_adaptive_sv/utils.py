import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def key2df(keys, delimeter="-"):
    key_df = pd.DataFrame(keys, columns=['key'])
    key_df['spk'] = key_df.key.apply(lambda x: x.split(delimeter)[0])
    key_df['session'] = key_df.key.apply(lambda x: x.split(delimeter)[1])
    key_df['label'] = key_df.groupby('spk').ngroup()
    key_df['idx'] = range(len(key_df))
    key_df = key_df.set_index('key')
    
    key_df['idx'] = range(len(key_df))
    id2idx = key_df.idx.to_dict()
    idx2id = {v:k for k,v in id2idx.items()}

    return key_df

def df2dict(key_df):
    key_df['idx'] = range(len(key_df))
    id2idx = key_df.idx.to_dict()
    idx2id = {v:k for k,v in id2idx.items()}

    return id2idx, idx2id

def get_id2idx(keys):
    key_df = key2df(keys)
    id2idx, idx2id = df2dict(key_df) 
    return id2idx

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

def compute_eer(scores, labels, verbose=True):
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
    eer = np.max([fprs[eer_idx], fnrs[eer_idx]])
    eer_fpr = fprs[eer_idx]
    eer_fnr = fnrs[eer_idx]
    eer_thresh = thres[eer_idx]
    
    if verbose:
        print("eer: {:.2f}%, fpr: {:.2f}%, fnr: {:.2f}%".format(
            eer*100, eer_fpr*100, eer_fnr*100))
        
    return eer, eer_fpr, eer_fnr, eer_thresh

def compare_value(val_a, val_b, mask=None, verbose=True):
    if mask is not None:
        val_a = val_a[mask]
        val_b = val_b[mask]
    assert len(val_a) == len(val_b)
    n = len(val_a)
    r_inc = np.count_nonzero(val_a < val_b) / n
    r_equal = np.count_nonzero(val_a == val_b) / n
    r_dec = np.count_nonzero(val_a > val_b) / n
    if verbose:
        print("inc:{:.2f}, equal:{:.2f}, dec:{:.2f}".format(r_inc, r_equal, r_dec))
    return r_inc, r_equal, r_dec

def plot_ROC(y_train_true, y_train_prob):
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
    import matplotlib.pyplot as plt
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label =True)
    sum_sensitivity_specificity_train = tpr_train + (1-fpr_train)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]
    best_fpr_train = fpr_train[best_threshold_id_train]
    best_tpr_train = tpr_train[best_threshold_id_train]
    y_train = y_train_prob > best_threshold

    cm_train = confusion_matrix(y_train_true, y_train)
    acc_train = accuracy_score(y_train_true, y_train)
    auc_train = roc_auc_score(y_train_true, y_train)

    print ('Train Accuracy: %s ' %acc_train)
    print ('Train AUC: %s ' %auc_train)
    print ('Train Confusion Matrix:')
    print (cm_train)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    curve1 = ax.plot(fpr_train, tpr_train)
    curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    ax.text(best_fpr_train, best_tpr_train, s = '(%.3f,%.3f)' %(best_fpr_train, best_tpr_train))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (Train), AUC = %.4f'%auc_train)
    plt.show()

    return best_threshold, fpr_train, tpr_train, thresholds_train

