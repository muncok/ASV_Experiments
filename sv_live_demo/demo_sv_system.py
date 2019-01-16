import librosa
# from pydub import AudioSegment
from demo_manage_audio import strip_audio
from demo_manage_audio import preprocess_audio
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from sklearn import preprocessing, cluster

def seg2wav(seg):
    wav_data = (np.array(seg.get_array_of_samples())
                / 32768.0).astype(np.float32)
    
    return wav_data

def zero_padding(data, in_len):
    padding_len = max(0, in_len - len(data))
    data = np.pad(data, (padding_len//2, padding_len - padding_len//2), "constant")
    
    return data

class demo_sv_system():
    def __init__(self, model, n_dims, feat_format, 
                 spk_models=None, lda_model=None):
        self.lda_model = pickle.load(open(lda_model, "rb")) if lda_model \
                                else None
        self.dct_filters = librosa.filters.dct(n_filters=n_dims, n_input=n_dims)
        self.model = model
        self.n_dims = n_dims
        self.feat_format = feat_format
        self.enrolled_wavs = dict()
        self.user_threshold = dict()
        self.enrolled_dvectors = dict()
        self.speaker_models =  pickle.load(open(spk_models, "rb")) if spk_models \
                                else None
            
    def enrol(self, wav, spk_name):
        feat = self._wav2feat(wav)
        dvector = self._extract_dvector(feat).squeeze()

        if spk_name not in self.enrolled_dvectors:
            self.enrolled_dvectors[spk_name] = [dvector]
            self.enrolled_wavs[spk_name] = [wav]
            self.user_threshold[spk_name] = 0.5
        else:
            self.enrolled_dvectors[spk_name] += [dvector]
            self.enrolled_wavs[spk_name] += [wav]
            
    def batch_enroll(self, wav_list, spk_name):
        feat_list = [self._wav2feat(wav) for wav in wav_list]
        feat = torch.stack(feat_list, dim=0)
        dvector = self._extract_dvector(feat).squeeze()

        assert spk_name not in self.enrolled_dvectors
        self.enrolled_dvectors[spk_name] = list(dvector)
        self.enrolled_wavs[spk_name] = wav_list
                   
    def _extract_dvector(self, feat):
        """
            feat: tensor
            dvector: ndarray
        """
        if torch.cuda.is_available():
            feat = feat.cuda()
            
        if feat.dim() == 2:
            feat = feat.unsqueeze(0).unsqueeze(0)
        elif feat.dim() == 3:
            feat = feat.unsqueeze(1)
    
        dvector = self.model.embed(feat).detach().cpu().numpy()
        if self.lda_model:
            dvector = self.lda_model.transform(dvector).astype(np.float32)
                
        return dvector
            
    def _wav2feat(self, wav):
        """
            extracting input feature from wav (mfcc, fbank)
        """
        wav_data = librosa.core.load(wav, sr=16000)[0]   
        feat = preprocess_audio(wav_data, n_mels=self.n_dims, 
                    dct_filters=self.dct_filters, in_feature=self.feat_format)
        
        return feat
    
    def _wav2dvector(self, wav):
        feat = self._wav2feat(wav)
        dvector = self._extract_dvector(feat)
        
        return dvector
        
    def init_enrollemnt(self):
        self.enrolled_feats = dict()
        self.enrolled_dvectors = dict()
                
    def multiple_dvector_verify(self, dvectors):
        pred_speakers = []
        scores = []
        for test_dvector in dvectors:
            score = []
            for k, v in self.enrolled_dvectors.items():
                v = np.array(v)
                score_ = F.cosine_similarity(torch.from_numpy(v).float(), 
                                             torch.from_numpy(test_dvector).float(), dim=1).numpy()
                score.append(np.mean(score_))
            pred_speaker = list(self.enrolled_dvectors.keys())[np.argmax(score)]
            pred_speakers.append(pred_speaker)
            scores.append(max(score))
            
        return pred_speakers, scores

    def verify(self, wav):
        """
            verify a input wav and output a verification result
            and rank-1 identification
        """
        feat = self._wav2feat(wav)
        test_dvector = self._extract_dvector(feat)
        score = []
        for k, v in self.enrolled_dvectors.items():
            v = np.array(v)
            score_ = F.cosine_similarity(torch.from_numpy(v).float(), 
                                         torch.from_numpy(test_dvector).float(), dim=1).numpy()
            score.append(np.mean(score_))
        pred_speaker = list(self.enrolled_dvectors.keys())[np.argmax(score)]
        
        result = "Reject"
        if max(score) > 0.704:
            result = "Accept"
        
        if max(score) > 0.827:
            result = "Enrolled"
            self.enrolled_dvectors[pred_speaker] += [test_dvector.flatten()]
            self.enrolled_wavs[pred_speaker] += [wav]
            
        return result, pred_speaker, max(score)