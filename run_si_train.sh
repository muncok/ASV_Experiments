#!/bin/bash

cmd="python sv_system/si_model_train.py \
-nep 80 -batch 128 -lrs 0.1 \
-dataset voxc_fbank_xvector1 \
-version 2 \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
