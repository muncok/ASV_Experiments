#!/bin/bash
export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


if [ $# != 2 ]; then
  echo "Usage: $0 plda_model_dir score_dir"
  echo "plda_model_dir contains plda, mean.vec, transform.mat"
  echo "score_dir contains kaldi_trial, feat.ark"
  exit 1;
fi

plda_dir=$1
feat_dir=$2
scores_dir=$2
trials=$2/kaldi_trial

mkdir -p $scores_dir/log
run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${plda_dir}/plda - |" \
    "ark:ivector-subtract-global-mean ${plda_dir}/mean.vec ark:${feat_dir}/feats.ark ark:- | transform-vec ${plda_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${plda_dir}/mean.vec ark:${feat_dir}/feats.ark ark:- | transform-vec ${plda_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;
