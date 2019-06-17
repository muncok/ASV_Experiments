#!/bin/bash
export KALDI_ROOT=/host/projects/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


if [ $# != 2 ]; then
  echo "Usage: $0 lda_model_dir score_dir"
  echo "lda_model_dir contains lda, mean.vec, transform.mat"
  echo "score_dir contains kaldi_trial, feat.ark"
  exit 1;
fi

lda_dir=$1
feat_dir=$2
scores_dir=$2
trials=$2/kaldi_trial

mkdir -p $scores_dir/log
run.pl $scores_dir/log/lda_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
  ivector-compute-dot-products - \
    "ark:ivector-subtract-global-mean ${lda_dir}/mean.vec ark:${feat_dir}/feats.ark ark:- | transform-vec ${lda_dir}/transform.mat ark:- ark:- | ivector-normalize-length --scaleup=false ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${lda_dir}/mean.vec ark:${feat_dir}/feats.ark ark:- | transform-vec ${lda_dir}/transform.mat ark:- ark:- | ivector-normalize-length --scaleup=false ark:- ark:- |" \
   $scores_dir/lda_scores || exit 1;
