Speaker Verification Experiments

# Unsupervised Speaker Adaptation

* Requirements
  * Python 3.6
  * pytorch-0.4.x
  * Pandas, tqdm ... (you can install them by pip or conda)

* Data
  * [Trial](https://drive.google.com/open?id=1dD7n4Vn56cdrb6A21C1sf-mDjPE0gDpm&authuser=muncok@dal.snu.ac.kr)
  * [embeddings](https://drive.google.com/open?id=1QIkKdmTi4sICGokWlwAy25cTGB2OEAT7&authuser=muncok@dal.snu.ac.kr)

* Run example
  `python run_trial.py -out_dir out_tmp -sv_mode base -trial_t random`

* Arguments

  ```
    usage: run_trial.py [-h] [-n_enr N_ENR] -out_dir OUT_DIR
                        [-n_process N_PROCESS] -sv_mode {base,inc} -trial_t
                        {random,sortedPos} [-ths_t {normal,extreme}] [-update]
                        [-incl_init]
  ```

  * -n_enr: number of initial enrollments.
  * -out_dir: directory where outputs are saved.
  * -n_process: number of processes which run experiments
  * -sv_mode: base=baseline, inc=incremental(proposed)
  * -trial_t: trial type {random sequence, session sorted}
  * -ths_t: threshold type; normal=found in validation, extreme=low threshold
  * -update: threshold update flag
  * -incl_init: always including init enrollment in enrollment queue
