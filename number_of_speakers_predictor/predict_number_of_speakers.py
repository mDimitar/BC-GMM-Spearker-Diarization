import numpy as np
import pickle
import librosa

def process_audio(file_path, n_mfcc=13, frame_length=2048, hop_length=512, sequence_length=4096):
    y, sr = librosa.load(file_path, sr=None)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)

    features = features.T

    if features.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - features.shape[0], n_mfcc))
        features = np.vstack((padding, features))
    else:
        features = features[:sequence_length, :]
        if features.shape[0] > sequence_length:
            print("Warning: Audio file is longer than the specified sequence length. Actual length:", features.shape[0])

    return features

with open('../binary_classifier/best_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

input_audio_file = "../test/test_audio_1.mp3"

X_input = process_audio(input_audio_file)

X_input = X_input.reshape(1, -1)

y_pred = clf.predict(X_input)

print(f"Predicted number of speakers: {y_pred[0]}")
