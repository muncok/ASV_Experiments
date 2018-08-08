#!/bin/bash

cmd="python sv_system/si_model_train.py \
-nep 80 -batch 256 \
-dataset voxc12_fbank \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
