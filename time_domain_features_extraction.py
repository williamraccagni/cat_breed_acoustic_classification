import librosa
import numpy as np



def time_domain_dataset(dataset_path : str, fn_dataset : list, frame_length : int, hop_length : int) -> dict:
    tdf_dataset = {}

    for x in fn_dataset:
        tdf_dataset[x[0]] = time_domain_features(dataset_path, x[1], frame_length, hop_length)

    return tdf_dataset



def time_domain_features(dataset_path : str, filename : str, frame_length : int, hop_length : int) -> np.array:

    y, _ = librosa.core.load(dataset_path + '/' + filename)

    ae = amplitude_envelope(signal=y, frame_length=frame_length, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    return np.concatenate((features_statistics(ae), features_statistics(zcr), features_statistics(rmse)), axis=0)



def amplitude_envelope(signal, frame_length : int, hop_length : int) -> np.array:
    return np.array([max(signal[i:i+frame_length]) for i in range(0, len(signal), hop_length)])



def features_statistics(features : np.array) -> np.array:
    return np.array([np.mean(features), np.median(features), np.std(features), np.amax(features), np.amin(features),
                     (np.std(features) / np.mean(features)) ])