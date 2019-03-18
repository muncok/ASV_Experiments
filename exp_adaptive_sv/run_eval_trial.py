import sys
import pickle
import numpy as np
import argparse
import pandas as pd


from tqdm import tqdm
from multiprocessing import Process, Manager
from batch_sv_system_utils import compute_plda_score, get_embeds, cosine_sim, compute_eer
from utils import get_id2idx

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=str)
parser.add_argument("--mode", type=str, choices=["online", "online_plda",
                    "batch_plda", "batch", "split_plda_proxy", "split_plda_sn"])
parser.add_argument("--eT", type=float, default=10)
parser.add_argument("--nTar", type=float, default=None, choices=[0.1, 1, 9, 99,
    None])
parser.add_argument("--out_file", type=str, default="./tmp/score_list.pkl")
args = parser.parse_args()

def score_s_norm(scores, enr_embeds, test_embeds, cohort_embeds):
    enr_cohort_scores = cosine_sim(init_enr_embeds, cohort_embeds)
    enr_mean = enr_cohort_scores.mean(1, keepdims=True)
    enr_std = enr_cohort_scores.std(1, keepdims=True)
    test_cohort_scores = cosine_sim(test_embeds, cohort_embeds)
    test_mean = test_cohort_scores.mean(1, keepdims=True).T
    test_std = test_cohort_scores.std(1, keepdims=True).T
    norm_scores = ((scores - enr_mean)/enr_std + (scores - test_mean)/test_std)/2

    return norm_scores

def online_score_wrapper(embeds, labels, score_q, eT=15):
    init_enr_embeds, test_embeds, cohort_embeds = embeds

    adapt_scores = cosine_sim(init_enr_embeds, test_embeds)
    adapt_norm_scores = score_s_norm(adapt_scores, init_enr_embeds,
            test_embeds, cohort_embeds)

    adapt_enr_idx = np.nonzero(adapt_norm_scores.mean(0) > eT)[0]
    enr_embeds = np.concatenate([init_enr_embeds, test_embeds[adapt_enr_idx]], axis=0)

    ### test trial
    test_scores = cosine_sim(enr_embeds, test_embeds)
    test_norm_scores = score_s_norm(test_scores, enr_embeds,
            test_embeds, cohort_embeds)

    ### reduce to online sv scores
    online_scores = []
    prev_t = 0
    adapt_times = np.append(adapt_enr_idx, len(test_trial[1]))
    for n_enr, adapt_t in enumerate(adapt_times):
        online_scores.append(test_norm_scores[:n_enr+1, prev_t:adapt_t+1])
        prev_t = adapt_t+1

    score_q.put((online_scores, labels, adapt_enr_idx))

def split_plda_proxy_score_wrapper(embeds, adapt_labels, labels, score_q, plda_model_dir, eT=10):
    plda_init_enr_embeds, plda_adapt_embeds, plda_test_embeds = embeds

    # adapt trial
    plda_adapt_scores = compute_plda_score(plda_init_enr_embeds, plda_adapt_embeds,
                                           plda_model_dir, all_pair=True)
    adapt_enr_idx = np.nonzero(plda_adapt_scores.mean(0) > eT)[0]
    plda_enr_embeds = np.concatenate([plda_init_enr_embeds, plda_adapt_embeds[adapt_enr_idx]],
                                     axis=0)
    # test trial
    plda_test_scores = compute_plda_score(plda_enr_embeds, plda_test_embeds, 
                                          plda_model_dir, all_pair=True)
    
    proxy_test_embeds = np.concatenate([plda_enr_embeds, cohort_embeds], axis=0)
    eval_proxy_labels = np.concatenate([np.ones(len(plda_enr_embeds)), np.zeros(len(cohort_embeds))])
    eval_proxy_scores = compute_plda_score(plda_enr_embeds, proxy_test_embeds, plda_model_dir)
    proxy_eers = []
    for i in range(0, len(eval_proxy_scores)):
        proxy_eers.append(compute_eer(eval_proxy_scores[i], eval_proxy_labels)[0])
    proxy_eer_sorted = np.argsort(proxy_eers)
    proxy_idx = proxy_eer_sorted.tolist()
    plda_proxy_test_scores = plda_test_scores[proxy_idx]
    
    # F-norm
    a = 1
    client_scores = eval_proxy_scores[:, :len(plda_enr_embeds)]
    client_mean = np.triu(client_scores, 1).mean()
    imp_scores = eval_proxy_scores[:, len(plda_enr_embeds):]
    imp_mean = imp_scores.mean()
    plda_test_norm_scores = (plda_test_scores-imp_mean)*(2*a/(client_mean - imp_mean)) + a
    plda_proxy_test_norm_scores = (plda_proxy_test_scores-imp_mean)*(2*a/(client_mean - imp_mean)) + a
    
    adapt_labels = np.array([1,1,1] + adapt_labels[adapt_enr_idx].tolist())

    score_q.put((plda_test_norm_scores, plda_proxy_test_norm_scores, labels, adapt_labels, proxy_idx))

def split_plda_sn_score_wrapper(embeds, adapt_labels, labels, score_q, plda_model_dir, eT=10):
    plda_init_enr_embeds, plda_adapt_embeds, plda_test_embeds = embeds

    # adapt trial
    plda_adapt_scores = compute_plda_score(plda_init_enr_embeds, plda_adapt_embeds,
                                           plda_model_dir, all_pair=True)
    adapt_enr_idx = np.nonzero(plda_adapt_scores.mean(0) > eT)[0]
    plda_enr_embeds = np.concatenate([plda_init_enr_embeds, plda_adapt_embeds[adapt_enr_idx]],
                                     axis=0)
    
    adapt_cohort_scores = compute_plda_score(plda_enr_embeds, cohort_embeds, plda_model_dir)
    if len(adapt_enr_idx) > 0:
        adapt_cohort_mu = adapt_cohort_scores[len(plda_init_enr_embeds):].mean(1)
        adapt_cohort_std = adapt_cohort_scores[len(plda_init_enr_embeds):].std(1)
        adapt_norm_scores = (plda_adapt_scores.mean(0)[adapt_enr_idx] - adapt_cohort_mu)/(adapt_cohort_std)
        norm_idx = [0,1,2] + (np.flip(np.argsort(adapt_norm_scores))+len(plda_init_enr_embeds)).tolist()
    else:
        norm_idx = [0,1,2]
    
    # test trial
    plda_test_scores = compute_plda_score(plda_enr_embeds, plda_test_embeds, 
                                          plda_model_dir, all_pair=True)
    plda_sn_test_scores = plda_test_scores[norm_idx]
    
    # F-norm
    a = 1
    client_scores = compute_plda_score(plda_enr_embeds, plda_enr_embeds, plda_model_dir)
    client_mean = np.triu(client_scores, 1).mean()
    imp_scores = adapt_cohort_scores
    imp_mean = imp_scores.mean()
    plda_test_norm_scores = (plda_test_scores-imp_mean)*(2*a/(client_mean - imp_mean)) + a
    plda_sn_test_norm_scores = (plda_sn_test_scores-imp_mean)*(2*a/(client_mean - imp_mean)) + a
    
    adapt_labels = np.array([1,1,1] + adapt_labels[adapt_enr_idx].tolist())

    score_q.put((plda_test_norm_scores, plda_sn_test_norm_scores, labels, adapt_labels, norm_idx))

def online_plda_score_wrapper(embeds, labels, score_q, plda_model_dir, eT=10):
    plda_init_enr_embeds, plda_test_embeds = embeds

    plda_adapt_scores = compute_plda_score(plda_init_enr_embeds,
            plda_test_embeds, plda_model_dir)
    adapt_enr_idx = np.nonzero(plda_adapt_scores.mean(0) > eT)[0]
    plda_enr_embeds = np.concatenate([plda_init_enr_embeds,
        plda_test_embeds[adapt_enr_idx]], axis=0)

    plda_test_scores = compute_plda_score(plda_enr_embeds, plda_test_embeds, plda_model_dir)

    plda_online_scores = []
    adapt_times = np.append(adapt_enr_idx, len(test_trial[1]))
    prev_t = 0
    for n_enr, adapt_t in enumerate(adapt_times):
        plda_online_scores.append(plda_test_scores[:n_enr+len(plda_init_enr_embeds),
                                                   prev_t:adapt_t+1])
        prev_t = adapt_t+1

    adapted_embed_scores = plda_adapt_scores.mean(0)[adapt_enr_idx]
    
    score_q.put((plda_online_scores, labels, adapted_embed_scores, enr_spk))

def batch_score_wrapper(embeds, labels, score_q, eT=10):
    init_enr_embeds, test_embeds = embeds

    adapt_scores = cosine_sim(init_enr_embeds, test_embeds)
    adapt_enr_idx = np.nonzero(adapt_scores.mean(0) > eT)[0]
    enr_embeds = np.concatenate([init_enr_embeds, test_embeds[adapt_enr_idx]], axis=0)

    test_scores = cosine_sim(enr_embeds, test_embeds)

    adapted_embed_scores = adapt_scores.mean(0)[adapt_enr_idx]
    adapt_labels = np.array([1,1,1] + labels[adapt_enr_idx].tolist())

    score_q.put((test_scores, labels, adapted_embed_scores, adapt_labels, enr_spk))

    
def batch_plda_score_wrapper(embeds, labels, score_q, plda_model_dir, eT=10):
    plda_init_enr_embeds, plda_test_embeds = embeds

    plda_adapt_scores = compute_plda_score(plda_init_enr_embeds,
                                           plda_test_embeds, plda_model_dir)
    adapt_enr_idx = np.nonzero(plda_adapt_scores.mean(0) > eT)[0]
    plda_enr_embeds = np.concatenate([plda_init_enr_embeds,
                                      plda_test_embeds[adapt_enr_idx]], axis=0)

    plda_test_scores = compute_plda_score(plda_enr_embeds, plda_test_embeds, plda_model_dir)
#     Z-norm
#     enr_cohort_scores = compute_plda_score(plda_enr_embeds, cohort_embeds, plda_model_dir)
#     enr_cohort_mu = enr_cohort_scores.mean(1, keepdims=True)
#     enr_cohort_std = enr_cohort_scores.std(1, keepdims=True)
#     plda_test_norm_scores = (plda_test_scores - enr_cohort_mu) / enr_cohort_std

    # F-norm
    a = 1
    client_scores = compute_plda_score(plda_enr_embeds, plda_enr_embeds, plda_model_dir)
    client_mean = np.triu(client_scores, 1).mean()
    imp_scores = compute_plda_score(plda_enr_embeds, cohort_embeds, plda_model_dir)
    imp_mean = imp_scores.mean()
    plda_test_norm_scores = (plda_test_scores-imp_mean)*(2*a/(client_mean - imp_mean)) + a

    adapted_embed_scores = plda_adapt_scores.mean(0)[adapt_enr_idx]
    adapt_labels = np.array([1,1,1] + labels[adapt_enr_idx].tolist())

#     score_q.put((plda_test_scores, labels, adapted_embed_scores, adapt_labels, enr_spk))
    score_q.put((plda_test_norm_scores, labels, adapted_embed_scores, adapt_labels, enr_spk))

if __name__=='__main__':
    trials = pickle.load(open(args.trial, "rb"))

    embed_dir = "embeddings/voxc2_fbank64_voxc2untied_embeds/"
    sv_embeds = np.load(embed_dir + "/sv_embeds.npy")
    keys = pickle.load(open(embed_dir + "/sv_keys.pkl", "rb"))
    id2idx = get_id2idx(keys)

    plda_embed_dir = "embeddings/voxc2_fbank64_voxc2untied_xvector/"
    plda_sv_embeds = np.load(plda_embed_dir + "/sv_embeds.npy")
    plda_model_dir = plda_embed_dir + "plda_train/"
    plda_keys = pickle.load(open(plda_embed_dir + "/sv_keys.pkl", "rb"))
    plda_id2idx = get_id2idx(plda_keys)
    
    cohort_embeds = np.load("trials/dev940_eval311/dev_cohort_embeds.npy")

    score_list = []
    n_parallel = 80
    eT = args.eT
    
#     trials = trials[0:500:10]
    print("total trial:{} with eT:{}".format(len(trials), eT))
    for i, idx in tqdm(enumerate(range(0, len(trials), n_parallel)),
                       ascii=True, desc="Jobs", file=sys.stdout,
                       total=len(trials)//n_parallel):
        procs = []
        manager = Manager()
        score_q = manager.Queue()

        for j, trial in enumerate(trials[idx:idx+n_parallel]):
            enr_spk, enr_ids, test_trial = trial
            print("enrolled speaker:{}".format(enr_spk))

            if args.nTar:
                n_target= test_trial.label.value_counts()[1]
                n_nonTarget = int(n_target*args.nTar)
                test_trial = pd.concat([test_trial[test_trial.label==1],
                                        test_trial[test_trial.label==0].sample(n=n_nonTarget)])

               # for  reproducibility 
#             test_trial = test_trial.sample(frac=1.0)
            test_trial = (np.array(test_trial.id), np.array(test_trial.label))

            if args.mode == "split_plda_proxy":
                ### batch plda trial
                adapt_len = 300
                adapt_trial = (test_trial[0][:adapt_len], test_trial[1][:adapt_len])
                test_trial = (test_trial[0][adapt_len:], test_trial[1][adapt_len:])
                plda_init_enr_embeds = get_embeds(enr_ids, plda_sv_embeds, 
                                                  plda_id2idx, norm=False)
                plda_adapt_embeds = get_embeds(adapt_trial[0], plda_sv_embeds,
                                               plda_id2idx, norm=False)
                plda_test_embeds = get_embeds(test_trial[0], plda_sv_embeds,
                                              plda_id2idx, norm=False)
                embeds = (plda_init_enr_embeds, plda_adapt_embeds, plda_test_embeds)
                proc = Process(target=split_plda_proxy_score_wrapper,
                        args=(embeds, adapt_trial[1], test_trial[1], score_q, plda_model_dir, eT))
            elif args.mode == "split_plda_sn":
                ### batch plda trial
                adapt_len = 300
                adapt_trial = (test_trial[0][:adapt_len], test_trial[1][:adapt_len])
                test_trial = (test_trial[0][adapt_len:], test_trial[1][adapt_len:])
                plda_init_enr_embeds = get_embeds(enr_ids, plda_sv_embeds, 
                                                  plda_id2idx, norm=False)
                plda_adapt_embeds = get_embeds(adapt_trial[0], plda_sv_embeds,
                                               plda_id2idx, norm=False)
                plda_test_embeds = get_embeds(test_trial[0], plda_sv_embeds,
                                              plda_id2idx, norm=False)
                embeds = (plda_init_enr_embeds, plda_adapt_embeds, plda_test_embeds)
                proc = Process(target=split_plda_sn_score_wrapper,
                        args=(embeds, adapt_trial[1], test_trial[1], score_q, plda_model_dir, eT))
            elif args.mode == "online_plda":
                ### online plda trial
                test_trial = (test_trial[0], test_trial[1])
                plda_init_enr_embeds = get_embeds(enr_ids, plda_sv_embeds,
                                                  plda_id2idx, norm=False)
                plda_test_embeds = get_embeds(test_trial[0], plda_sv_embeds, 
                                              plda_id2idx, norm=False)
                embeds = (plda_init_enr_embeds, plda_test_embeds)
                proc = Process(target=online_plda_score_wrapper,
                        args=(embeds, test_trial[1], score_q, plda_model_dir, eT))
            elif args.mode == "batch_plda":
                ### online plda trial
                test_trial = (test_trial[0], test_trial[1])
                plda_init_enr_embeds = get_embeds(enr_ids, plda_sv_embeds, 
                                                  plda_id2idx, norm=False)
                plda_test_embeds = get_embeds(test_trial[0], plda_sv_embeds,
                                              plda_id2idx, norm=False)
                embeds = (plda_init_enr_embeds, plda_test_embeds)
                proc = Process(target=batch_plda_score_wrapper,
                        args=(embeds, test_trial[1], score_q, plda_model_dir, eT))
            elif args.mode == "batch":
                ### online plda trial
                test_trial = (test_trial[0], test_trial[1])
                init_enr_embeds = get_embeds(enr_ids, sv_embeds, 
                                                  id2idx, norm=True)
                test_embeds = get_embeds(test_trial[0], sv_embeds,
                                              id2idx, norm=True)
                embeds = (init_enr_embeds, test_embeds)
                proc = Process(target=batch_score_wrapper,
                        args=(embeds, test_trial[1], score_q, eT))
            elif args.mode == "online":
                ### online score_norm  trial
                test_trial = (test_trial[0], test_trial[1])
                init_enr_embeds = get_embeds(enr_ids, sv_embeds, id2idx, norm=False)
                test_embeds = get_embeds(test_trial[0], sv_embeds, id2idx, norm=False)
                embeds = (init_enr_embeds, test_embeds)
                proc = Process(target=online_score_wrapper,
                        args=(embeds, test_trial[1], score_q, eT))
            else:
                raise NotImplementedError

            procs.append(proc)
            proc.start()
                
        for p in procs:
            score_list.append(score_q.get())
            p.join()

    out_file = args.out_file
    pickle.dump(score_list, open(out_file, "wb"))
