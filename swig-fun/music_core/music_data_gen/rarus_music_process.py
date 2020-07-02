import librosa
import numpy as np
import os
import requests
from src.utils.music_utils.mp3_load_util import load_audio
from audioread.maddec import UnsupportedError


"""
Notes:
The samplerate is set to a default mp3 samplerate : 22050.
Every sample will be resamples to match this.
NB. THE MAIN FUNCTION CREATES A PICKLE IF GIVEN "save=True"
IF NOT GIVEN, IT WILL ONLY RETURN A DATAFRAME
A wrapped processs that can be run in parallel by dask
"""


def process_wrapper(id):
    # Generate mp3 from api
    tmp_file_name = extract_file_from_api(id)
    if tmp_file_name is not None:

        try:
            result = process_api_file(tmp_file_name, id)
            os.remove(tmp_file_name)
        except UnsupportedError:
            os.remove(tmp_file_name)
            result = "Kunne ikke fjerne fil"

    else:
        result = "Fejl i udtræk fra api"

    return(result)


# Load tempfile from api
def extract_file_from_api(file_id):
    try:
        file_api_req = requests.get(
            "http://play.api.247e.com/portals/1458/track-samples/" +
            str(file_id)).json()
        url = file_api_req['httpOptions'][0]['links'][0]['href']
        filename = 'temp'+str(file_id)+'.mp3'
        open(filename, 'wb').write(requests.get(url).content)
    except:
        return None
    return(filename)


# Methods that we need to run
def process_api_file(file_id_path, track_id):
    # Loading the specific file
    y, sr = load_audio(
        file_id_path,
        offset=0.0,
        duration=28.0,
        dtype=np.float32)
    feature_list = ["mfcc", "H", "P", "spectral_contrast"]
    # Setting track id and tempo
    tempo = librosa.beat.tempo(y)
    # Extract features
    cov_features, order_rec = extract_features_from_list(y, sr, feature_list)
    # Calculate the covariance matrix
    the_stack = np.vstack(cov_features)
    cov = np.cov(the_stack)
    # Calculating the means of each variable
    means = generate_means(cov_features)
    # Wrapping in a dataframe
    return(tempo, cov, means, order_rec)


# Extract features defined in list
def extract_features_from_list(y, sr, feature_list):

    # Function that returns the features in a list
    # outputs two lists, one for calculating
    # the covariance and one for the means
    # AND one for the order

    cov_features = []
    order_rec = []

    if "chroma_stft" in feature_list:
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        cov_features.append(chroma_stft)
        order_rec.append("chroma_stft")
    if "chrome_cqt" in feature_list:
        chrome_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        cov_features.append(chrome_cqt)
        order_rec.append("chrome_cqt")
    if "chroma_cens" in feature_list:
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        cov_features.append(chroma_cens)
        order_rec.append("chroma_cens")
    if "melspectrogram" in feature_list:
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        cov_features.append(melspectrogram)
        order_rec.append("melspectrogram")
    if "rms" in feature_list:
        rms = librosa.feature.rms(y=y)
        cov_features.append(rms)
        order_rec.append("rms")
    if "spectral_centroid" in feature_list:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        cov_features.append(spectral_centroid)
        order_rec.append("spectral_centroid")
    if "spectral_bandwidth" in feature_list:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        cov_features.append(spectral_bandwidth)
        order_rec.append("spectral_bandwidth")
    if "spectral_contrast" in feature_list:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        cov_features.append(spectral_contrast)
        order_rec.append("spectral_contrast")
    if "spectral_flatness" in feature_list:
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        cov_features.append(spectral_flatness)
        order_rec.append("spectral_flatness")
    if "spectral_rolloff" in feature_list:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        cov_features.append(spectral_rolloff)
        order_rec.append("spectral_rolloff")
    if "poly_features" in feature_list:
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        cov_features.append(poly_features)
        order_rec.append("poly_features")
    if "tonnetz" in feature_list:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        cov_features.append(tonnetz)
        order_rec.append("tonnetz")
    if "tempogram" in feature_list:
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        cov_features.append(tempogram)
        order_rec.append("tempogram")
    if "fourier_tempogram" in feature_list:
        fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr)
        cov_features.append(fourier_tempogram)
        order_rec.append("fourier_tempogram")
    if "zero_crossing_rate" in feature_list:
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        cov_features.append(zero_crossing_rate)
        order_rec.append("zero_crossing_rate")
    if "mfcc" in feature_list:
        mfcc = librosa.feature.mfcc(y=y, sr=sr)[0:7]
        cov_features.append(mfcc)
        order_rec.append("mfcc")
    if "H" in feature_list:
        H = librosa.decompose.hpss(
            librosa.feature.spectral_centroid(y=y, sr=sr))[0]
        cov_features.append(H)
        order_rec.append("H")
    if "P" in feature_list:
        P = librosa.decompose.hpss(
            librosa.feature.spectral_centroid(y=y, sr=sr))[1]
        cov_features.append(P)
        order_rec.append("P")
    else:
        print("something went wrong with extraction")
    return(cov_features, order_rec)


# Generate means of each vector
def generate_means(list_of_np_features):
    # Empty list
    list_of_means = []
    # Calc all means as list
    for elem in list_of_np_features:
        for vec in elem:
            list_of_means.append(vec.mean())
    return(list_of_means)
