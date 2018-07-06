#!/bin/bash

cmd="python sv_system/si_model_train.py \
-nep 100 -batch 128 -lrs 0.1 \
-dataset voxc_fbank_xvector \
-model ResNet34_v3 -loss angular \
-version 2 -suffix si_set_vad_angular \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
