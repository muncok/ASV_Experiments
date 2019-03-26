import os, sys
import pickle
import numpy as np
import argparse
import pandas as pd


from tqdm import tqdm
from multiprocessing import Process, Manager
from batch_sv_system import get_embeds
from utils import key2df, df2dict, compute_eer, get_id2idx
from ioffe_plda.verifier import Verifier

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=str, required=True)
parser.add_argument("--n_jobs", type=int, required=True)
parser.add_argument("--out_dir", type=str, default="./tmp/")
args = parser.parse_args()


def py_plda_wrapper():
    scores = py_plda_model.score_avg(enr_emb, test_emb)
    score_label_q.put((scores[0], label))


if __name__=='__main__':
    
    trials = pd.read_pickle(args.trial)

    embed_dir = "embeddings/voxc2_fbank64_voxc2untied_xvector/"
    sv_embeds = np.load(embed_dir+"ln_lda_sv_embeds.npy")
    sv_keys = pickle.load(open(embed_dir + "/sv_keys.pkl", "rb"))
    sv_id2idx = get_id2idx(sv_keys)
    
    enr_embeds = get_embeds(trials.enr_id, sv_embeds, sv_id2idx, norm=False)
    test_embeds = get_embeds(trials.test_id, sv_embeds, sv_id2idx, norm=False)
    labels = trials.label.tolist()
    
    py_plda_model = Verifier()
    py_plda_model = Verifier(pickle.load(open("py_plda_model_ln_lda.pkl", "rb")))

    score_label_list = []
    n_jobs = args.n_jobs
    
    iter_len = len(trials)
    for i, idx in tqdm(enumerate(range(0, iter_len, n_jobs)),
                       ascii=True, desc="Jobs", file=sys.stdout,
                       total=iter_len//n_jobs):
        procs = []
        manager = Manager()
        score_label_q = manager.Queue()

        for enr_emb, test_emb, label in zip(enr_embeds[idx:idx+n_jobs], 
                                            test_embeds[idx:idx+n_jobs], 
                                            labels[idx:idx+n_jobs]):
            
            proc = Process(target=py_plda_wrapper,
                    args=())

            procs.append(proc)
            proc.start()

        for p in procs:
            score_label_list.append(score_label_q.get())
            p.join()

    save_dir = args.out_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pickle.dump(score_label_list, open(save_dir + "/score_label_list.pkl", "wb"))
