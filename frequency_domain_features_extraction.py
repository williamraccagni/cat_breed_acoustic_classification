import librosa
import numpy as np



def frequency_domain_dataset(dataset_path : str, fn_dataset : list, fft_length : int, hop_length : int) -> dict:
    fdf_dataset = {}

    for x in fn_dataset:
        fdf_dataset[x[0]] = frequency_domain_features(dataset_path, x[1], fft_length, hop_length)

    return fdf_dataset



def frequency_domain_features(dataset_path : str, filename : str, fft_length : int, hop_length : int) -> np.array:

    # windowing hann?

    y, sr = librosa.core.load(dataset_path + '/' + filename)

    spec_cent = features_statistics(librosa.feature.spectral_centroid(y=y, sr=sr,
                                                                                     n_fft=fft_length,
                                                                                     hop_length=hop_length))
    spec_flux = features_statistics(librosa.onset.onset_strength(y=y, sr=sr))
    rolloff = features_statistics(librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                                                 n_fft=fft_length,
                                                                                 hop_length=hop_length))
    mfcc = flat_mfcc_stat(y=y, sr=sr, fft_length=fft_length, hop_length=hop_length, n_mfcc=13)

    return np.concatenate((spec_cent, spec_flux, rolloff, mfcc), axis=0)



def flat_mfcc_stat(y, sr, fft_length : int, hop_length : int, n_mfcc : int):

    lib_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=fft_length, hop_length=hop_length, n_mfcc=n_mfcc)

    result = features_statistics(lib_mfcc[0])

    for index in range(1,n_mfcc):
        result = np.concatenate((result, features_statistics(lib_mfcc[index])), axis=0)

    return result



def features_statistics(features : np.array) -> np.array:
    return np.array([np.mean(features), np.median(features), np.std(features), np.amax(features), np.amin(features),
                     (np.std(features) / np.mean(features)) ])
