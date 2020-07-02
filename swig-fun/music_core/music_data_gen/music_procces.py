

import librosa
import numpy as np
import io
import requests
import dask.dataframe as dd

if __name__ == "__main__":
    from mp3_load_util import load_audio
else:
    from .mp3_load_util import load_audio

"""
Notes:
The samplerate is set to a default mp3 samplerate : 22050.
Every sample will be resamples to match this.
NB. THE MAIN FUNCTION CREATES A PICKLE IF GIVEN "save=True"
IF NOT GIVEN, IT WILL ONLY RETURN A DATAFRAME
A wrapped processs that can be run in parallel by dask
"""

# Load tempfile from api
def extract_mp3_from_247_api(id):
    try:
        file_api_req = requests.get(
            "http://play.api.247e.com/portals/1458/track-samples/" +
            str(id)).json()
        url = file_api_req['httpOptions'][0]['links'][0]['href']
        return io.BytesIO(requests.get(url).content)
    except:
        return None

feature_mappings = {
    "chroma_stft":librosa.feature.chroma_stft,
    "chrome_cqt": librosa.feature.chroma_cqt,
    "chroma_cens": librosa.feature.chroma_cens,
    "melspectrogram": librosa.feature.melspectrogram,
    "rms": lambda y, sr: librosa.feature.rms(y=y),
    "spectral_centroid": librosa.feature.spectral_centroid,
    "spectral_bandwidth": librosa.feature.spectral_bandwidth,
    "spectral_contrast": lambda y, sr: librosa.feature.spectral_contrast(y=y, sr=sr),
    "spectral_flatness": lambda y, sr: librosa.feature.spectral_flatness(y=y),
    "spectral_rolloff": librosa.feature.spectral_rolloff,
    "poly_features": librosa.feature.poly_features,
    "tonnetz": librosa.feature.tonnetz,
    "tempogram": librosa.feature.tempogram,
    "fourier_tempogram": librosa.feature.fourier_tempogram,
    "zero_crossing_rate": lambda y, sr: librosa.feature.zero_crossing_rate(y),
    "mfcc": lambda y, sr: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8),
    "H": lambda y, sr: librosa.decompose.hpss(librosa.feature.spectral_centroid(y=y, sr=sr))[0],
    "P": lambda y, sr: librosa.decompose.hpss(librosa.feature.spectral_centroid(y=y, sr=sr))[1]
}

feature_keys = feature_mappings.keys()

# Extract features defined in list
def extract_features(y, sr, feature_list=feature_keys):

    # Function that returns the features in a list
    # outputs two lists, one for calculating
    # the covariance and one for the means
    # AND one for the order

    return {k: feature_mappings[k](y, sr) for k in feature_list}


# Methods that we need to run
def process_audio(fp, track_id, feature_list=feature_mappings.keys(), offset=0.0, duration=None):
    # Loading the specific file
    y, sr = load_audio(
        fp,
        offset=offset,
        duration=duration,
        dtype=np.float32)
    # Setting track id and tempo
    track_identification = track_id
    tempo = librosa.beat.tempo(y)
    # Extract features
    features = extract_features(y, sr, feature_list)

    lower = lambda C: C[np.tril_indices(C.shape[0])] if C.ndim > 1 else C

    # Calculate the covariance matrix
    gauss_func = lambda f: (np.mean(f, axis=1), lower(np.cov(f)))

    return {

        "track_id": track_identification,
        "tempo": tempo,
        **{k: gauss_func(f) for k, f in features.items()}
    }

def process_wrapper(id, feature_list, sample_api=extract_mp3_from_247_api, **kwargs):
    # Generate mp3 from api
    mp3_stream = sample_api(id)
    try:
        return process_audio(mp3_stream, id, feature_list, **kwargs)
    except:
        return None

def music_data_generator(id_series, feature_list, save=False, **kwargs):

    ddata = dd.from_array(id_series, npartitions=10)

    # Applying the process
    new_data = ddata.map_partitions(
        lambda df: df.apply(
            lambda row: process_wrapper(row, feature_list, **kwargs))).compute(
                                                scheduler='processes')
    # Ugly return statement, give save as True
    # to save as pkl file
    return new_data

if __name__=="__main__":
    filepath ="C:\\Users\\m87184\\projects\\nuudai_recomendation_engine\\fma_metadata\\13_-_FRESH.mp3"
    with open(filepath, 'rb') as fid:
        res = process_audio(fid, 64)

    a = 0
