import os
import pickle
import pandas as pd
import numpy as np
from enum import Enum

class Mode(Enum):
    base = 0
    inc = 1
    inc_update = 2
    inc_neg = 3
    inc_update_neg = 4
    
def analyze_trace(trs):
    results = []
    for tr in trs:
        if len(tr) == 4:
            trial_idxs, scores, label, pred = tr
        elif len(tr) == 5:
            trial_idxs, scores, label, pred, enroll_pred = tr
        else:
            raise NotImplementedError

        result = dict()
        result["fpr"] = round(np.count_nonzero((pred == 1) & (label == 0)) / np.count_nonzero(label == 0), 4)
        try:
            result["fnr"] = round(np.count_nonzero((pred == 0) & (label == 1)) / np.count_nonzero(label == 1), 4)
        except ZeroDivisionError:
            # for ood trial
            result["fnr"] = 0
        result["error"] = round(np.count_nonzero(pred != label) / len(label), 4)

        if len(tr) == 5:
            result["n_adt_enroll"] = np.count_nonzero(enroll_pred != -1)
             # 1: correctly enrolled, 0: incorrectly enrolled, -1: not enrolle
            if np.count_nonzero(enroll_pred != -1) == 0:
                result["enr_error"] = 0
            else:
                result["enr_error"] = round(np.count_nonzero(enroll_pred == 0) \
                                            / np.count_nonzero(enroll_pred != -1), 4)
        results.append(result)

    return results

def compute_metrics(results):
        mean_error = np.mean([dic['error'] for dic in results])
        mean_fpr = np.mean([dic['fpr'] for dic in results])
        mean_fnr = np.mean([dic['fnr'] for dic in results])
        
        return mean_error, mean_fpr, mean_fnr
    
def print_result(result_dir):
    modes = os.listdir(result_dir)
    setting = result_dir.split("/")[-2]
    modes = sorted(modes, key=lambda x: Mode[x].value)

    print(setting)
    for mode in modes:
        mode_dir = os.path.join(result_dir, mode)
        with open(mode_dir + "/trace.pkl", "rb") as f:
            trace = pickle.load(f)
        meta_info = pd.read_pickle(mode_dir + "/meta_info_df.pkl")
        results = {"adapt":[], "test":[], "ood":[], "test_s_err":[], 'ood_s_err':[]}
        for tr in trace:
            adapt_result, test_result, ood_result = analyze_trace(tr)
            results["adapt"].append(adapt_result)
            results["test"].append(test_result)
            results["ood"].append(ood_result)
            results["test_s_err"].append(meta_info['test_s_err'])
            results["ood_s_err"].append(meta_info['ood_s_err'])
            
        for trial_type in ["adapt", "test", "ood"]:
            mean_error, mean_fpr, mean_fnr = compute_metrics(results[trial_type])
            msg = ("[{:20s}]: {:.4f}(err), {:.4f}(fpr), {:.4f}(fnr)".format(
                "/".join([mode, trial_type]), mean_error, mean_fpr, mean_fnr))
            if trial_type != "adapt":
                err = np.mean(results[trial_type + "_s_err"])
                msg += " {:.4f}(s_err)".format(err)
            print(msg)
        print("-"*80)
    print("="*80)
    
def summary_result(result_dir):
    modes = os.listdir(result_dir)
    setting = result_dir.rstrip("/").split("/")[-1]
    modes = sorted(modes, key=lambda x: Mode[x].value)

    summary = []
    errors = {'adapt':[], 'test':[], 'test_s_err':[], 'ood':[], 'ood_s_err':[]}
    for mode in modes:
        mode_dir = os.path.join(result_dir, mode)
        with open(mode_dir + "/trace.pkl", "rb") as f:
            trace = pickle.load(f)
        meta_info = pd.read_pickle(mode_dir + "/meta_info_df.pkl")
        results = {"adapt":[], "test":[], "ood":[], "test_s_err":[], 'ood_s_err':[]}
        # accumulating over traces
        for tr in trace:
            adapt_result, test_result, ood_result = analyze_trace(tr)
            results["adapt"].append(adapt_result)
            results["test"].append(test_result)
            results["ood"].append(ood_result)
            results["test_s_err"].append(meta_info['test_s_err'])
            results["ood_s_err"].append(meta_info['ood_s_err'])
            
        # averaging over accumulated data
        for trial_type in ["adapt", "test", "ood"]:
            mean_error, mean_fpr, mean_fnr = compute_metrics(results[trial_type])
            errors[trial_type] += [mean_error]
            if trial_type != "adapt":
                errors[trial_type+"_s_err"] += [np.mean(results[trial_type + "_s_err"])]
                
    # Aggerating over modes        
    for trial_type in ["adapt", "test", "ood"]:
        record = [setting, trial_type] + errors[trial_type] 
        if trial_type != "adapt":
            record += errors[trial_type + "_s_err"]
        else:
            record += [0., 0., 0.]
        summary.append(record)
            
    return modes, summary