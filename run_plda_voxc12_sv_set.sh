#!/bin/bash

export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=1

data_dir=$1
train_dir=$data_dir/plda_train
test_dir=$data_dir/plda_test
trials=trials/voxc12_trial/voxceleb1_trials_sv
#trials=exp_embedding/DDA/voxc12_trial_sv
scores_dir=$data_dir/plda_scores

if [ $stage -le 1 ]; then

mkdir -p $train_dir
mkdir -p $test_dir
#cp trials/voxc12_trial/spk2utt $train_dir
#cp trials/voxc12_trial/utt2spk $train_dir

# change the /train-feat to /test-feat
./kaldi_utils/feat2ark.py -key_in $1/test_feat/key.pkl -embed_in $1/test_feat/feat.npy -output $train_dir
./kaldi_utils/feat2ark.py -key_in $1/test_feat/key.pkl -embed_in $1/test_feat/feat.npy -output $test_dir

# compute dvector mean for following global mean subtraction
ivector-mean ark:$train_dir/feats.ark $train_dir/mean.vec || exit 1;

fi

#if [ $stage -le 2 ]; then

## This script uses LDA to decrease the dimensionality prior to PLDA.
#lda_dim=200
#run.pl $data_dir/train/log/lda.log \
#ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#"ark:ivector-subtract-global-mean scp:$train_dir/feats.scp ark:- |" \
#ark:$train_dir/utt2spk $train_dir/transform.mat || exit 1;

#fi


if [ $stage -le 3 ]; then

#compute plda model

run.pl $train_dir/log/plda.log \
    ivector-compute-plda ark:$train_dir/spk2utt \
    "ark:ivector-subtract-global-mean ark:${train_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    $train_dir/plda || exit 1;
fi

if [ $stage -le 4 ]; then

# plda scoring
mkdir -p $scores_dir/log

run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${train_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec ark:${test_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

# comput eer and minDCFs
eer=`compute-eer <(kaldi_utils/prepare_for_eer.py $trials $scores_dir/plda_scores) 2> /dev/null`
mindcf1=`kaldi_utils/compute_min_dcf.py --p-target 0.01 $scores_dir/plda_scores $trials 2> /dev/null`
mindcf2=`kaldi_utils/compute_min_dcf.py --p-target 0.001 $scores_dir/plda_scores $trials 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

fi
