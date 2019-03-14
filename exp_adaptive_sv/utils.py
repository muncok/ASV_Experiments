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

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return eer

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

