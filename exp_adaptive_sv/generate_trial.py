import os, sys
import pickle
import numpy as np
import argparse
import pandas as pd


from tqdm import tqdm
from multiprocessing import Process, Manager
from batch_sv_system_utils import compute_plda_score, get_embeds, cosine_sim
from utils import get_id2idx

parser = argparse.ArgumentParser()
parser.add_argument("--n_enr", type=int, default=3)
parser.add_argument("--r_enr", type=int, default=10)
parser.add_argument("--mode", type=str, default='random', choices=['random', 'hard'])
parser.add_argument("--set", type=str, default='eval', choices=['dev', 'eval'])
parser.add_argument("--out_dir", type=str, default="./tmp/")
args = parser.parse_args()


def gen_hard_trial_wrapper():
    scores = compute_plda_score(enr_embeds, nonTarget_embeds, plda_model_dir)

    eval_trials = []
    for i, idx in enumerate(range(0, len(enr_uttrs), args.n_enr)):
        enr_uttrs_ = enr_uttrs[idx:idx+args.n_enr]
        hard_neg_trials = np.argsort(
            scores[idx:idx+args.r_enr].mean(0), axis=0)[-len(test_target_uttrs)*9:].ravel()
        test_nonTarget_uttrs = nonTarget_uttrs.iloc[hard_neg_trials]
        test_trial = pd.concat([target_uttrs, test_nonTarget_uttrs])
        # for reproducibility
        test_trial = test_trial.sample(frac=1.0)
        eval_trials += [(eval_spk+"_"+str(i).zfill(2), np.array(enr_uttrs_.id), test_trial)]
                      
    trial_q.put(eval_trials)
    
def gen_trial_wrapper():
    eval_trials = []
    for i, idx in enumerate(range(0, len(enr_uttrs), args.n_enr)):
        enr_uttrs_ = enr_uttrs[idx:idx+args.n_enr]
        test_nonTarget_uttrs = nonTarget_uttrs.sample(n=9*len(test_target_uttrs))
        test_trial = pd.concat([target_uttrs, test_nonTarget_uttrs])
        # for reproducibility
        test_trial = test_trial.sample(frac=1.0)
        eval_trials += [(eval_spk+"_"+str(i).zfill(2), np.array(enr_uttrs_.id), test_trial)]
                      
    trial_q.put(eval_trials)

if __name__=='__main__':
    voxc1_df = pd.read_csv("/dataset/SV_sets/voxceleb1/dataframes/voxc1.csv")
    spk_uttr_stat = voxc1_df.spk.value_counts()
    
    if args.set == "eval":
        eval_spks = spk_uttr_stat[spk_uttr_stat >= 150].index.tolist()
        eval_uttrs = voxc1_df[voxc1_df.spk.isin(eval_spks)][['id', 'spk', 'gender', 'session']]
    elif args.set == "dev":
        eval_spks = spk_uttr_stat[(spk_uttr_stat < 150)].index.tolist()
        eval_uttrs = voxc1_df[voxc1_df.spk.isin(eval_spks)][['id', 'spk', 'gender', 'session']]

    plda_embed_dir = "embeddings/voxc2_fbank64_voxc2untied_xvector/"
    plda_sv_embeds = np.load(plda_embed_dir + "/sv_embeds.npy")
    plda_model_dir = plda_embed_dir + "plda_train/"
    plda_keys = pickle.load(open(plda_embed_dir + "/sv_keys.pkl", "rb"))
    plda_id2idx = get_id2idx(plda_keys)

    trial_list = []
    n_parallel = 80
    for i, idx in tqdm(enumerate(range(0, len(eval_spks), n_parallel)),
                       ascii=True, desc="Jobs", file=sys.stdout,
                       total=len(eval_spks)//n_parallel):
        procs = []
        manager = Manager()
        trial_q = manager.Queue()

        for j, eval_spk in enumerate(eval_spks[idx:idx+n_parallel]):
            target_uttrs = eval_uttrs[eval_uttrs.spk == eval_spk]
            nonTarget_uttrs = eval_uttrs[eval_uttrs.spk != eval_spk]
            target_uttrs.loc[:, 'label'] = 1
            nonTarget_uttrs.loc[:, 'label'] = 0
            
#             enr_uttrs = target_uttrs.sample(n=args.n_enr*args.r_enr)
            enr_uttrs = target_uttrs.groupby("session", group_keys=False).apply(lambda x: x.sample(n=3, replace=True))
            enr_uttrs = enr_uttrs.drop(columns="session")
            test_target_uttrs = target_uttrs.drop(index=enr_uttrs.index)
            test_target_uttrs = test_target_uttrs.drop(columns="session")
            enr_embeds = get_embeds(enr_uttrs.id, plda_sv_embeds, plda_id2idx)
            nonTarget_embeds = get_embeds(nonTarget_uttrs.id, plda_sv_embeds, plda_id2idx)
            
            uttrs = (target_uttrs, nonTarget_uttrs)
            embeds = (enr_embeds, nonTarget_embeds)
        
            if args.mode == "random":
                proc = Process(target=gen_trial_wrapper,
                        args=())
            elif args.mode == "hard":
                proc = Process(target=gen_hard_trial_wrapper,
                        args=())
                
            procs.append(proc)
            proc.start()

        for p in procs:
            trial_list.extend(trial_q.get())
            p.join()

    save_dir = args.out_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pickle.dump(trial_list, open(save_dir + "/trials.pkl", "wb"))    
