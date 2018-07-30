#!/bin/bash

cmd="python sv_system/si_model_train.py \
-nep 80 -batch 256 -lrs 0.1 \
-dataset voxc12_fbank_xvector \
-version 2 \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
