#!/bin/bash

cmd="python sv_system/extract_dvector_voxc12.py \
-batch 128 \
-dataset voxc12_fbank_xvector \
-inFm fbank -inFr 800 -spFr 800  \
-cuda \
$@"
echo $cmd
$cmd
