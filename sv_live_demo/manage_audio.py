import librosa
import numpy as np
import scipy
import torch

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def preprocess_audio(data, n_mels, dct_filters, in_feature="mfcc"):
    ## DCT Part
    if in_feature == "mfcc":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        ## appending deltas
        # data_delta = librosa.feature.delta(data)
        # data_delta2 = librosa.feature.delta(data, order=2)
        # data = np.stack([data, data_delta, data_delta2], axis=0)
        data = torch.from_numpy(data)
    elif in_feature == "fbank":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = data.astype(np.float32).transpose()
        data = torch.from_numpy(data)
        mean = data.mean(0) # along time dimension
        data.add_(-mean)
        # std = data.std(0)
        # data.div_(std)
    elif in_feature == "fft":
        data = fft_audio(data, 0.025, 0.010)
    else:
        raise NotImplementedError
    return data  # dims:3, with no channel dimension.

def fft_audio(data, window_size, window_stride):
    n_fft = 480
    # n_fft = int(16000*window_size)
    win_length = int(16000* window_size)
    hop_length = int(16000* window_stride)
    # STFT
    D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=windows['hamming'])
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect) # (freq, time)
    spect = torch.FloatTensor(spect.T) # (time, freq)
    # normalization
    mean = spect.mean(0) # over time dim
    std = spect.std(0)
    spect.add_(-mean)
    spect.div_(std)
    return spect

def strip_audio(x, frame_length=1024, hop_length=256, rms_ths=0.2):
    # compute energy
    rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length)[0]
    rms_ratio = rmse/rmse.max()

    active_frames = np.nonzero(rms_ratio > rms_ths)[0]
    assert len(active_frames) > 0, "there is no voice part in the wav"

    # strip continous active part
    s_sample = librosa.frames_to_samples(active_frames[0], hop_length=hop_length)[0]
    e_sample = librosa.frames_to_samples(active_frames[-1], hop_length=hop_length)[0]

	# plot the rmse on the wavelet of x
    # frames = range(len(energy))
	# import matplot.pyplot as plt
    # energy = np.array([
        # sum(abs(x[i:i+frame_length]**2))
        # for i in range(0, len(x), hop_length)
    # ])
    # t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    # librosa.display.waveplot(x, sr=sr, alpha=0.4)
    # plt.plot(t, energy/energy.max(), 'r--')             # normalized for visualization
    # plt.plot(t[:len(rmse)], rmse/rmse.max(), color='g') # normalized for visualization
    # plt.legend(('Energy', 'RMSE'))

    return x[s_sample:e_sample]
