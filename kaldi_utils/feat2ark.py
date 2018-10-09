#!/usr/bin/env python
import kaldi_io
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-key_in', type=str, help='input key pickle file')
parser.add_argument('-embed_in', type=str, help='input embed  npy file')
parser.add_argument('-output', type=str, help='output directory where ark and scp are saved')

args = parser.parse_args()
# print(args)

# dvector_dict = pickle.load(open(args.input, "rb"))
keys = pickle.load(open(args.key_in, "rb"))
embeds = np.load(args.embed_in)
ark_scp_output='ark:| copy-vector ark:- \
ark,scp:{output}/feats.ark,{output}/feats.scp'.format(output=args.output)
with kaldi_io.open_or_fd(ark_scp_output, "wb") as f:
    for key, vec in zip(keys, embeds):
        kaldi_io.write_vec_flt(f, vec.squeeze(), key=str(key))
