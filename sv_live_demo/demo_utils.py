import os
import pandas as pd
import torch

from demo_tdnnModel import tdnn_xvector_untied
from demo_ResNet34 import ResNet34_cr
from demo_sv_system import demo_sv_system


def load_model(path):
    config = dict(
        loss="softmax",
        gpu_no=[0], no_cuda=False,
        input_dim=40,
        feat_format='fbank'
    )
    
    model = tdnn_xvector_untied(config, n_labels=1759)
    model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
    model.eval()
        
    return model

def load_cr_model(path):
    config = dict(
        loss="softmax",
        gpu_no=[0], no_cuda=False,
        input_dim=40,
        feat_format='fbank'
    )
    
    model = ResNet34_cr(config, inplanes=64, n_labels=10)
    model.load(path)
    model.eval()
        
    return model
        
def load_dataset():
    def append_dir(file):
        return os.path.join(dirpath, file)
    
    wav_files = []
    for dirpath, dirnames, filenames in os.walk("/dataset/SV_sets/kor_commands/v2/wav"):
        if len(filenames) == 0: continue
        wav_files += list(map(append_dir, filenames))

    df = pd.DataFrame(wav_files, columns=['wav'])
    spk = df.wav.apply(lambda x: x.split('/')[-3])
    cmd = df.wav.apply(lambda x: x.split('/')[-2])
    df['spk'] = spk
    df['sent'] = cmd
  
    return df

def split_dataset(df, init_spk, init_cmd, n_per_sent):
    enroll_df = df[(df.spk.isin(init_spk)) & (df.sent.isin(init_cmd))]\
                .groupby(['spk','sent'], as_index=False, group_keys=False)\
                .apply(lambda x: x.sample(n=n_per_sent))
    test_df = df.drop(index=enroll_df.index)

    return enroll_df, test_df

def load_sv_system(model):
    sv_system = demo_sv_system(model, spk_models=None, lda_model=None, n_dims=40,
                          feat_format='fbank')
    
    return sv_system