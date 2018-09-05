#!/bin/bash

cmd="python sv_system/sv_score_voxc.py \
-batch 128 \
-dataset voxc_fbank \
-model ResNet34_v1 \
-loss softmax \
-inFm fbank -inFr 200 -spFr 200  \
-cuda \
$1 $2"
#-input_file models/compare_train_methods/voxc_fbank_xvector/ResNet34/fbank_800f_800f_angular_si_set/e00.pt -s_epoch 10
echo $cmd
$cmd
