import librosa
from demo_manage_audio import preprocess_audio
import numpy as np
import torch
import torch.nn.functional as F

class cr_system():
    def __init__(self, model, n_dims=40, feat_format='fbank'):
        self.classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.dct_filters = librosa.filters.dct(n_filters=n_dims, n_input=n_dims)
        self.model = model
        self.n_dims = n_dims
        self.feat_format = feat_format
        
    def _get_score(self, feat):
        """
            dvector: ndarray
        """
        
        if torch.cuda.is_available():
            feat = feat.cuda()
            
        if feat.dim() == 2:
            feat = feat.unsqueeze(0).unsqueeze(0)
        
        dvector = self.model.forward(feat).detach().cpu().numpy()
        
        return dvector
    
    def _wav2feat(self, wav):
        """
            extracting input feature from wav (mfcc, fbank)
        """
        wav_data = librosa.core.load(wav, sr=16000)[0]
        feat = preprocess_audio(wav_data, n_mels=self.n_dims, 
                    dct_filters=self.dct_filters, in_feature=self.feat_format)
        
        return feat

    def recog(self, wav):
        """
            verify a input wav and output a verification result
            and rank-1 identification
        """
        feat = self._wav2feat(wav)
        score = self._get_score(feat)
        pred_class = self.classes[np.argmax(score)]

        return pred_class
            
