import numpy as np
import librosa
import os


def extract_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def process_audio(file_path, n_mfcc=13, frame_length=2048, hop_length=512, sequence_length=4096):
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features with the specified frame and hop length
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)

    features = features.T

    # Pad or truncate the features to ensure a consistent length
    if features.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - features.shape[0], n_mfcc))
        features = np.vstack((padding, features))
    else:
        features = features[:sequence_length, :]
        if features.shape[0] > sequence_length:
            print("Warning: Audio file is longer than the specified sequence length. Actual length:", features.shape[0])

    return features


def process_and_save_batches(directory_path, n_mfcc=13, sequence_length=4096, batch_size=512):
    files = [f for f in os.listdir(directory_path) if f.endswith('.mp3')]
    total_batches = len(files) // batch_size + (1 if len(files) % batch_size > 0 else 0)

    for batch_index in range(total_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(files))
        batch_files = files[start_index:end_index]

        X_batch = []
        y_batch = []

        for filename in batch_files:
            file_path = os.path.join(directory_path, filename)
            features = process_audio(file_path, n_mfcc=n_mfcc, sequence_length=sequence_length)
            X_batch.append(features)

            num_speakers = int(filename.split('_')[1])
            y_batch.append(num_speakers)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        combined = {'X': X_batch, 'y': y_batch}
        np.save(f'./batches/batch_{batch_index}.npy', combined)
        print(f"Batch {batch_index} saved")


if __name__ == '__main__':
    data_directory_path = '../DiarizationDataset'
    n_mfcc = 13
    sequence_length = 4096
    batch_size = 64

    process_and_save_batches(data_directory_path, n_mfcc=n_mfcc, sequence_length=sequence_length, batch_size=batch_size)
