#!/usr/bin/env python
import kaldi_io
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, help='input dvec pickle file')
parser.add_argument('-output', type=str, help='output directory where ark and scp are saved')

args = parser.parse_args()

dvector_dict = pickle.load(open(args.input, "rb"))
ark_scp_output='ark:| copy-vector ark:- \
ark,scp:{output}/feats.ark,{output}/feats.scp'.format(output=args.output)
with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
    for key, vec in dvector_dict.items():
        kaldi_io.write_vec_flt(f, vec.squeeze(), key=str(key))
