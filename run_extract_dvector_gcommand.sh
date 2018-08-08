#!/bin/bash

cmd="python sv_system/extract_dvector_gcommand.py \
-batch 128 \
-dataset gcommand_fbank \
-inFm fbank -inFr 80 -spFr 80  \
-cuda \
$@"
echo $cmd
$cmd
