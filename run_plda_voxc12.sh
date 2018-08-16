#!/bin/bash
export KALDI_ROOT=`pwd`/../kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

stage=0

. kaldi_voxc12/cmd.sh

data_dir=$1
train_dir=$data_dir/train
test_dir=$data_dir/test
trials=kaldi_voxc12/voxceleb1_trials_sv
scores_dir=$data_dir/plda_scores

if [ $stage -le 0 ]; then

mkdir -p $train_dir
mkdir -p $test_dir
cp kaldi_voxc12/data/train_combined_no_sil/spk2utt $train_dir

# place dvecoter at data-dvector/~
./numpy2ark.py -input $data_dir/voxc_train_dvectors.pkl -output $train_dir || exit 1;
./numpy2ark.py -input $data_dir/voxc_test_dvectors.pkl -output $test_dir || exit 1;

# compute dvector mean for following global mean subtraction
ivector-mean ark:$train_dir/feats.ark $train_dir/mean.vec || exit 1;

fi

if [ $stage -le 1 ]; then

# compute plda model
$train_cmd $train_dir/log/plda.log \
    ivector-compute-plda ark:$train_dir/spk2utt \
    "ark:ivector-subtract-global-mean ark:${train_dir}/feats.ark ark:- | ivector-normalize-length ark:- ark:- |" \
    $train_dir/plda || exit 1;

mkdir -p $scores_dir/log

fi

if [ $stage -le 2 ]; then

# plda scoring
$train_cmd $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${train_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec scp:${test_dir}/feats.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${train_dir}/mean.vec scp:${test_dir}/feats.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

# comput eer and minDCFs
eer=`compute-eer <(kaldi_voxc/local/prepare_for_eer.py $trials $scores_dir/plda_scores) 2> /dev/null`
mindcf1=`kaldi_voxc/sid/compute_min_dcf.py --p-target 0.01 $scores_dir/plda_scores $trials 2> /dev/null`
mindcf2=`kaldi_voxc/sid/compute_min_dcf.py --p-target 0.001 $scores_dir/plda_scores $trials 2> /dev/null`
echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

fi

