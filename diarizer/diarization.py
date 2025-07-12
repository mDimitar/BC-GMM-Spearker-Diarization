import numpy as np
import pickle
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def process_audio(file_path, n_mfcc=13, frame_length=2048, hop_length=512, sequence_length=4096):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    features = features.T

    if features.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - features.shape[0], n_mfcc))
        features = np.vstack((padding, features))
    else:
        features = features[:sequence_length, :]
        if features.shape[0] > sequence_length:
            print(f"Audio file is longer than the specified sequence length. Actual length: {features.shape[0]}")

    return features


def voice_activity_detection(wav_data, frame_rate):
    hop_length = int(16000 / frame_rate)
    if hop_length <= 0:
        raise ValueError("Invalid frameRate resulting in hop_length <= 0")

    ste = librosa.feature.rms(y=wav_data, hop_length=hop_length).T
    thresh = 0.1 * (np.percentile(ste, 97.5) + 9 * np.percentile(ste, 2.5))
    return (ste > thresh).astype('bool')


def train_gmm(wav_file, frame_rate, seg_len, voice_activity_detection, num_mix):
    wav_data, _ = librosa.load(wav_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=wav_data, sr=16000, n_mfcc=40, hop_length=int(16000 / frame_rate)).T
    voice_activity_detection = np.reshape(voice_activity_detection, (len(voice_activity_detection),))
    if mfcc.shape[0] > voice_activity_detection.shape[0]:
        voice_activity_detection = np.hstack((voice_activity_detection,
                                              np.zeros(mfcc.shape[0] - voice_activity_detection.shape[0]).astype(
                                                  'bool'))).astype('bool')
    elif mfcc.shape[0] < voice_activity_detection.shape[0]:
        voice_activity_detection = voice_activity_detection[:mfcc.shape[0]]
    mfcc = mfcc[voice_activity_detection, :]
    print("Training GMM..")
    GMM = GaussianMixture(n_components=num_mix, covariance_type='diag').fit(mfcc)
    seg_likes = []
    seg_size = frame_rate * seg_len
    for segI in range(int(np.ceil(float(mfcc.shape[0]) / (frame_rate * seg_len)))):
        start_i = segI * seg_size
        end_i = (segI + 1) * seg_size
        if end_i > mfcc.shape[0]:
            end_i = mfcc.shape[0] - 1
        if end_i == start_i:
            break
        seg = mfcc[start_i:end_i, :]
        comp_likes = np.sum(GMM.predict_proba(seg), 0)
        seg_likes.append(comp_likes / seg.shape[0])
    print("Training Done")

    return np.asarray(seg_likes)


def segment_frame(clust, seg_len, frame_rate, num_frames):
    frame_clust = np.zeros(num_frames)
    for clust_i in range(len(clust) - 1):
        frame_clust[clust_i * seg_len * frame_rate:(clust_i + 1) * seg_len * frame_rate] = clust[clust_i] * np.ones(
            seg_len * frame_rate)
    frame_clust[(clust_i + 1) * seg_len * frame_rate:] = clust[clust_i + 1] * np.ones(
        num_frames - (clust_i + 1) * seg_len * frame_rate)
    return frame_clust


def speakerdiarisationdf(hyp, frame_rate, wav_file):
    audio_name = []
    starttime = []
    endtime = []
    speakerlabel = []

    spkr_change_points = np.where(hyp[:-1] != hyp[1:])[0]
    if len(spkr_change_points) == 0:
        audio_name.append(wav_file.split('/')[-1].split('.')[0] + ".wav")
        starttime.append(0)
        duration = librosa.get_duration(path=wav_file)
        endtime.append(duration)
        speakerlabel.append("Speaker 0")
    else:
        if spkr_change_points[0] != 0 and hyp[0] != -1:
            spkr_change_points = np.concatenate(([0], spkr_change_points))
        spkr_labels = []
        for spkr_homo_seg_i in range(len(spkr_change_points)):
            spkr_labels.append(hyp[spkr_change_points[spkr_homo_seg_i] + 1])
        for spkrI, spkr in enumerate(spkr_labels[:-1]):
            if spkr != -1:
                audio_name.append(wav_file.split('/')[-1].split('.')[0] + ".wav")
                starttime.append((spkr_change_points[spkrI] + 1) / float(frame_rate))
                endtime.append((spkr_change_points[spkrI + 1] - spkr_change_points[spkrI]) / float(frame_rate))
                speakerlabel.append("Speaker " + str(int(spkr)))
        if spkr_labels[-1] != -1:
            audio_name.append(wav_file.split('/')[-1].split('.')[0] + ".wav")
            starttime.append(spkr_change_points[-1] / float(frame_rate))
            endtime.append((len(hyp) - spkr_change_points[-1]) / float(frame_rate))
            speakerlabel.append("Speaker " + str(int(spkr_labels[-1])))

    speakerdf = pd.DataFrame(
        {"Audio": audio_name, "starttime": starttime, "endtime": endtime, "speakerlabel": speakerlabel})

    spdatafinal = pd.DataFrame(columns=['Audio', 'SpeakerLabel', 'StartTime', 'EndTime'])
    i = 0
    k = 0
    spfind = ""
    stime = ""
    etime = ""
    for row in speakerdf.itertuples():
        if i == 0:
            spfind = row.speakerlabel
            stime = row.starttime
        else:
            if spfind == row.speakerlabel:
                etime = row.endtime
            else:
                spdatafinal.loc[k] = [wav_file.split('/')[-1].split('.')[0] + ".wav", spfind, stime, row.starttime]
                k += 1
                spfind = row.speakerlabel
                stime = row.starttime
                etime = row.endtime
        i += 1
    duration = librosa.get_duration(path=wav_file)
    spdatafinal.loc[k] = [wav_file.split('/')[-1].split('.')[0] + ".wav", spfind, stime, duration]
    return spdatafinal

if __name__ == "__main__":
    with open('../binary_classifier/best_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)

    input_audio_file = "../test/K-15 - Ako sakas da sednam.mp3"
    X_input = process_audio(input_audio_file)

    if X_input is not None:
        X_input = X_input.reshape(1, -1)
        y_pred = clf.predict(X_input)
        num_speakers = y_pred[0]
        print(f"Predicted number of speakers: {num_speakers}")

        y, sr = librosa.load(input_audio_file, sr=16000)
        vad = voice_activity_detection(y, 50)

        if num_speakers == 1:
            frameClust = np.zeros(y.shape[0])
        else:
            clusterset = train_gmm(input_audio_file, 50, 3, vad, num_speakers)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clusterset)
            X_normalized = normalize(X_scaled)

            cluster = AgglomerativeClustering(n_clusters=num_speakers, linkage='average')
            labels = cluster.fit_predict(X_normalized)

            frameClust = segment_frame(labels, 3, 50, y.shape[0])

        diarization_df = speakerdiarisationdf(frameClust, 50, input_audio_file)
        print(f"Diarization results:\n{diarization_df}")
    else:
        print("Failed to process the audio file.")
