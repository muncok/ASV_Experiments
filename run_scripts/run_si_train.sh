#!/bin/bash

cmd="python sv_system/si_model_train.py \
-nep 100 -batch 128 -lrs 0.1 \
-dataset voxc_fbank \
-model ResNet34_v3 \
-loss softmax -version 2 -suffix si_set  \
-inFm fbank -inFr 800 -spFr 800  \
-cuda
$@"
#-input_file models/compare_train_methods/voxc_fbank_xvector/ResNet34/fbank_800f_800f_angular_si_set/e00.pt -s_epoch 10
echo $cmd
$cmd
