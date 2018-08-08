#!/bin/bash

cmd="python sv_system/extract_dvector_voxc1.py \
-batch 128 \
-dataset voxc1_fbank \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
