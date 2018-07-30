#!/bin/bash

cmd="python sv_system/sv_score_voxc.py \
-batch 128 \
-dataset voxc_fbank_xvector \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
