import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
import os
import json
from typing import List, Tuple, Dict, Optional
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')


class VBxDiarization:
    def __init__(self, frame_rate=100, seg_len=1.5, num_mix=16):
        self.frame_rate = frame_rate
        self.seg_len = seg_len
        self.num_mix = num_mix
        self.speaker_models = {}

    def extract_features(self, wav_file: str, voice_activity_detection: np.ndarray = None):
        """Extract MFCC features from audio file"""
        wav_data, _ = librosa.load(wav_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=wav_data, sr=16000, n_mfcc=40,
                                    hop_length=int(16000 / self.frame_rate)).T

        if voice_activity_detection is not None:
            voice_activity_detection = np.reshape(voice_activity_detection,
                                                  (len(voice_activity_detection),))

            if mfcc.shape[0] > voice_activity_detection.shape[0]:
                voice_activity_detection = np.hstack((
                    voice_activity_detection,
                    np.zeros(mfcc.shape[0] - voice_activity_detection.shape[0]).astype('bool')
                )).astype('bool')
            elif mfcc.shape[0] < voice_activity_detection.shape[0]:
                voice_activity_detection = voice_activity_detection[:mfcc.shape[0]]

            mfcc = mfcc[voice_activity_detection, :]

        return mfcc

    def train_speaker_gmm(self, mfcc_features: np.ndarray, speaker_id: str):
        """Train GMM for a specific speaker"""
        print(f"Training GMM for speaker {speaker_id}...")
        gmm = GaussianMixture(n_components=self.num_mix, covariance_type='diag',
                              random_state=42, max_iter=100)
        gmm.fit(mfcc_features)
        self.speaker_models[speaker_id] = gmm
        print(f"Training done for speaker {speaker_id}")
        return gmm

    def extract_segment_embeddings(self, mfcc_features: np.ndarray, gmm_model):
        """Extract segment-level embeddings using trained GMM"""
        seg_likes = []
        seg_size = int(self.frame_rate * self.seg_len)

        for segI in range(int(np.ceil(float(mfcc_features.shape[0]) / seg_size))):
            start_i = segI * seg_size
            end_i = min((segI + 1) * seg_size, mfcc_features.shape[0])

            if end_i <= start_i:
                break

            seg = mfcc_features[start_i:end_i, :]
            if seg.shape[0] > 0:
                comp_likes = np.sum(gmm_model.predict_proba(seg), 0)
                seg_likes.append(comp_likes / seg.shape[0])

        return np.asarray(seg_likes)

    def vbx_clustering(self, embeddings: np.ndarray, num_speakers: int = None,
                       max_speakers: int = 10):
        """VBx clustering algorithm"""
        print("Performing VBx clustering...")

        if embeddings.shape[0] < 2:
            return np.array([0])

        distances = []
        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                if i != j:
                    dist = cosine(embeddings[i], embeddings[j])
                else:
                    dist = 0.0
                row.append(dist)
            distances.append(row)

        distances = np.array(distances)

        linkage_matrix = linkage(distances, method='ward')

        if num_speakers is None:
            best_score = -np.inf
            best_labels = None

            for n_clusters in range(1, min(max_speakers + 1, len(embeddings) + 1)):
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

                if n_clusters > 1:
                    score = self._compute_clustering_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                else:
                    if best_labels is None:
                        best_labels = labels

            labels = best_labels
        else:
            labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust') - 1

        labels = self._vbx_refinement(embeddings, labels)

        print(f"VBx clustering completed. Found {len(np.unique(labels))} speakers")
        return labels

    def _compute_clustering_score(self, embeddings: np.ndarray, labels: np.ndarray):
        """Compute clustering quality score"""
        if len(np.unique(labels)) <= 1:
            return -1.0

        score = 0.0
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) <= 1:
                continue

            cluster_embeddings = embeddings[cluster_mask]
            cluster_center = np.mean(cluster_embeddings, axis=0)

            intra_dist = np.mean([cosine(emb, cluster_center) for emb in cluster_embeddings])

            other_embeddings = embeddings[~cluster_mask]
            if len(other_embeddings) > 0:
                inter_dist = np.mean([cosine(cluster_center, emb) for emb in other_embeddings])
                score += (inter_dist - intra_dist)

        return score

    def _vbx_refinement(self, embeddings: np.ndarray, initial_labels: np.ndarray,
                        max_iterations: int = 10):
        """VBx iterative refinement"""
        labels = initial_labels.copy()

        for iteration in range(max_iterations):
            old_labels = labels.copy()

            cluster_centers = {}
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_centers[cluster_id] = np.mean(embeddings[cluster_mask], axis=0)

            for i, embedding in enumerate(embeddings):
                best_cluster = labels[i]
                best_distance = cosine(embedding, cluster_centers[best_cluster])

                for cluster_id, center in cluster_centers.items():
                    distance = cosine(embedding, center)
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = cluster_id

                labels[i] = best_cluster

            if np.array_equal(labels, old_labels):
                print(f"VBx converged after {iteration + 1} iterations")
                break

        return labels

    def _align_labels(self, true_labels: np.ndarray, pred_labels: np.ndarray):
        """
        Align predicted labels with true labels using Hungarian algorithm
        to handle the permutation problem in clustering evaluation
        """

        true_unique = np.unique(true_labels)
        pred_unique = np.unique(pred_labels)

        conf_matrix = confusion_matrix(true_labels, pred_labels)

        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        label_mapping = {}
        for pred_idx, true_idx in zip(col_ind, row_ind):
            if pred_idx < len(pred_unique) and true_idx < len(true_unique):
                label_mapping[pred_unique[pred_idx]] = true_unique[true_idx]

        aligned_pred_labels = np.array([label_mapping.get(label, label) for label in pred_labels])

        return aligned_pred_labels

    def evaluate_diarization(self, true_labels: np.ndarray, pred_labels: np.ndarray,
                             detailed: bool = True) -> Dict:
        """
        Evaluate diarization performance using multiple metrics

        Args:
            true_labels: Ground truth speaker labels
            pred_labels: Predicted speaker labels
            detailed: Whether to return detailed per-class metrics

        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating diarization performance...")

        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        aligned_pred_labels = self._align_labels(true_labels, pred_labels)

        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        precision_macro = precision_score(true_labels, aligned_pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, aligned_pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, aligned_pred_labels, average='macro', zero_division=0)

        precision_micro = precision_score(true_labels, aligned_pred_labels, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, aligned_pred_labels, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, aligned_pred_labels, average='micro', zero_division=0)

        precision_weighted = precision_score(true_labels, aligned_pred_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, aligned_pred_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, aligned_pred_labels, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'num_true_speakers': len(np.unique(true_labels)),
            'num_pred_speakers': len(np.unique(aligned_pred_labels))
        }

        if detailed:

            precision_per_class = precision_score(true_labels, aligned_pred_labels, average=None, zero_division=0)
            recall_per_class = recall_score(true_labels, aligned_pred_labels, average=None, zero_division=0)
            f1_per_class = f1_score(true_labels, aligned_pred_labels, average=None, zero_division=0)

            metrics['per_class_metrics'] = {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist()
            }

            conf_matrix = confusion_matrix(true_labels, aligned_pred_labels)
            metrics['confusion_matrix'] = conf_matrix.tolist()

            try:
                class_report = classification_report(true_labels, aligned_pred_labels,
                                                     output_dict=True, zero_division=0)
                metrics['classification_report'] = class_report
            except:
                pass

        return metrics

    @staticmethod
    def parse_time_to_seconds(time_str: str) -> float:
        """Parse time string like '1:23s' or '00:09s' to seconds"""
        time_str = time_str.strip().replace('s', '')

        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)

        return float(time_str)

    def convert_ground_truth_to_segments(self, ground_truth_json):
        """Convert your ground truth format to VBx format"""

        converted_data = {}

        for item in ground_truth_json:
            audio_file = item["audio_file"]
            segments = []

            speakers_dict = item["diarization"]["speakers"][0]

            for speaker_name, time_ranges in speakers_dict.items():
                speaker_id = int(speaker_name.split('_')[1]) - 1

                for time_range in time_ranges:
                    try:
                        start_str, end_str = time_range.split(' - ')
                        start_time = self.parse_time_to_seconds(start_str)
                        end_time = self.parse_time_to_seconds(end_str)

                        if start_time < end_time:
                            segments.append((start_time, end_time, speaker_id))
                    except:
                        continue

            segments.sort(key=lambda x: x[0])
            converted_data[audio_file] = segments

        return converted_data

    def print_evaluation_results(self, metrics: Dict):
        """Print evaluation results in a formatted way"""
        print("\n" + "=" * 60)
        print("DIARIZATION EVALUATION RESULTS")
        print("=" * 60)

        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Number of True Speakers:  {metrics['num_true_speakers']}")
        print(f"Number of Pred Speakers:  {metrics['num_pred_speakers']}")

        print("\n" + "-" * 40)
        print("MACRO AVERAGED METRICS")
        print("-" * 40)
        print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")

        print("\n" + "-" * 40)
        print("MICRO AVERAGED METRICS")
        print("-" * 40)
        print(f"Precision (Micro):  {metrics['precision_micro']:.4f}")
        print(f"Recall (Micro):     {metrics['recall_micro']:.4f}")
        print(f"F1-Score (Micro):   {metrics['f1_micro']:.4f}")

        print("\n" + "-" * 40)
        print("WEIGHTED AVERAGED METRICS")
        print("-" * 40)
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")

        if 'per_class_metrics' in metrics:
            print("\n" + "-" * 40)
            print("PER-CLASS METRICS")
            print("-" * 40)
            for i, (p, r, f) in enumerate(zip(metrics['per_class_metrics']['precision'],
                                              metrics['per_class_metrics']['recall'],
                                              metrics['per_class_metrics']['f1'])):
                print(f"Speaker {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

        if 'confusion_matrix' in metrics:
            print("\n" + "-" * 40)
            print("CONFUSION MATRIX")
            print("-" * 40)
            conf_matrix = np.array(metrics['confusion_matrix'])
            print(conf_matrix)

        print("=" * 60)

    def train_on_dataset(self, dataset_path: str, annotation_file: str = None):
        """Train the model on your dataset"""
        print("Training VBx model on dataset...")

        annotations = {}
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

        audio_files = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.mp3', '.flac'))]

        for audio_file in audio_files:
            file_path = os.path.join(dataset_path, audio_file)
            print(f"Processing {audio_file}...")

            mfcc_features = self.extract_features(file_path)

            if audio_file in annotations:
                speaker_segments = annotations[audio_file]
                for speaker_id, segments in speaker_segments.items():
                    speaker_features = []
                    for start_time, end_time in segments:
                        start_frame = int(start_time * self.frame_rate)
                        end_frame = int(end_time * self.frame_rate)
                        speaker_features.append(mfcc_features[start_frame:end_frame])

                    if speaker_features:
                        combined_features = np.vstack(speaker_features)
                        self.train_speaker_gmm(combined_features, speaker_id)
            else:

                general_gmm = GaussianMixture(n_components=self.num_mix,
                                              covariance_type='diag', random_state=42)
                mfcc_features = mfcc_features.astype(np.float64)
                general_gmm.fit(mfcc_features)
                self.speaker_models['general'] = general_gmm

        print("Training completed!")

    def diarize(self, wav_file: str, voice_activity_detection: np.ndarray = None,
                num_speakers: int = None) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        """Perform speaker diarization on audio file"""
        print(f"Performing diarization on {wav_file}...")

        mfcc_features = self.extract_features(wav_file, voice_activity_detection)

        if 'general' in self.speaker_models:
            gmm_model = self.speaker_models['general']
        else:
            gmm_model = list(self.speaker_models.values())[0]

        embeddings = self.extract_segment_embeddings(mfcc_features, gmm_model)

        speaker_labels = self.vbx_clustering(embeddings, num_speakers)

        segments = []
        seg_duration = self.seg_len

        for i, speaker_id in enumerate(speaker_labels):
            start_time = i * seg_duration
            end_time = (i + 1) * seg_duration
            segments.append((start_time, end_time, int(speaker_id)))

        print("Diarization completed!")
        return speaker_labels, segments

    def diarize_and_evaluate(self, wav_file: str, true_segments: List[Tuple[float, float, int]] = None,
                             voice_activity_detection: np.ndarray = None,
                             num_speakers: int = None) -> Tuple[
        Optional[Dict], np.ndarray, List[Tuple[float, float, int]]]:
        """
        Perform diarization and optionally evaluate against ground truth

        Args:
            wav_file: Path to audio file
            true_segments: Optional ground truth segments in format [(start, end, speaker_id), ...]
            voice_activity_detection: Optional VAD array
            num_speakers: Optional number of speakers

        Returns:
            Tuple of (evaluation_metrics or None, predicted_labels, predicted_segments)
        """

        pred_labels, pred_segments = self.diarize(wav_file, voice_activity_detection, num_speakers)

        if true_segments is None:
            print("No ground truth provided - returning predictions only")
            return None, pred_labels, pred_segments

        audio_duration = librosa.get_duration(filename=wav_file)
        num_segments = int(np.ceil(audio_duration / self.seg_len))
        true_labels = np.zeros(num_segments, dtype=int)

        for start_time, end_time, speaker_id in true_segments:
            start_seg = int(start_time / self.seg_len)
            end_seg = int(end_time / self.seg_len)
            start_seg = max(0, start_seg)
            end_seg = min(num_segments, end_seg)
            true_labels[start_seg:end_seg] = speaker_id

        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        metrics = self.evaluate_diarization(true_labels, pred_labels)

        return metrics, pred_labels, pred_segments

    def analyze_diarization_quality(self, pred_labels: np.ndarray, embeddings: np.ndarray = None) -> Dict:
        """
        Analyze diarization quality using unsupervised metrics when no ground truth is available

        Args:
            pred_labels: Predicted speaker labels
            embeddings: Optional segment embeddings for internal clustering metrics

        Returns:
            Dictionary containing quality metrics
        """
        print("Analyzing diarization quality without ground truth...")

        num_speakers = len(np.unique(pred_labels))
        num_segments = len(pred_labels)

        quality_metrics = {
            'num_speakers_detected': num_speakers,
            'num_segments': num_segments,
            'segments_per_speaker': num_segments / num_speakers if num_speakers > 0 else 0
        }

        speaker_counts = {}
        for speaker_id in np.unique(pred_labels):
            count = np.sum(pred_labels == speaker_id)
            speaker_counts[f'speaker_{speaker_id}_segments'] = count
            speaker_counts[f'speaker_{speaker_id}_percentage'] = (count / num_segments) * 100

        quality_metrics.update(speaker_counts)

        speaker_changes = np.sum(pred_labels[1:] != pred_labels[:-1])
        quality_metrics['speaker_changes'] = speaker_changes
        quality_metrics['avg_segment_length'] = num_segments / (
                speaker_changes + 1) if speaker_changes >= 0 else num_segments

        if embeddings is not None and len(embeddings) == len(pred_labels):
            try:

                intra_cluster_distances = []
                inter_cluster_distances = []

                for speaker_id in np.unique(pred_labels):
                    speaker_mask = pred_labels == speaker_id
                    if np.sum(speaker_mask) <= 1:
                        continue

                    speaker_embeddings = embeddings[speaker_mask]
                    speaker_center = np.mean(speaker_embeddings, axis=0)

                    for emb in speaker_embeddings:
                        intra_cluster_distances.append(cosine(emb, speaker_center))

                    other_embeddings = embeddings[~speaker_mask]
                    for emb in other_embeddings:
                        inter_cluster_distances.append(cosine(emb, speaker_center))

                if intra_cluster_distances and inter_cluster_distances:
                    avg_intra = np.mean(intra_cluster_distances)
                    avg_inter = np.mean(inter_cluster_distances)
                    quality_metrics['avg_intra_cluster_distance'] = avg_intra
                    quality_metrics['avg_inter_cluster_distance'] = avg_inter
                    quality_metrics['cluster_separation_score'] = avg_inter - avg_intra

            except Exception as e:
                print(f"Could not compute clustering metrics: {e}")

        return quality_metrics

    def print_quality_analysis(self, quality_metrics: Dict):
        """Print quality analysis results"""
        print("\n" + "=" * 60)
        print("DIARIZATION QUALITY ANALYSIS (NO GROUND TRUTH)")
        print("=" * 60)

        print(f"Number of Speakers Detected: {quality_metrics['num_speakers_detected']}")
        print(f"Total Segments: {quality_metrics['num_segments']}")
        print(f"Average Segments per Speaker: {quality_metrics['segments_per_speaker']:.2f}")
        print(f"Speaker Changes: {quality_metrics['speaker_changes']}")
        print(f"Average Segment Length: {quality_metrics['avg_segment_length']:.2f}")

        print("\n" + "-" * 40)
        print("SPEAKER DISTRIBUTION")
        print("-" * 40)

        for key, value in quality_metrics.items():
            if 'speaker_' in key and '_segments' in key:
                speaker_id = key.split('_')[1]
                percentage_key = f'speaker_{speaker_id}_percentage'
                if percentage_key in quality_metrics:
                    print(f"Speaker {speaker_id}: {value} segments ({quality_metrics[percentage_key]:.1f}%)")

        if 'cluster_separation_score' in quality_metrics:
            print("\n" + "-" * 40)
            print("CLUSTERING QUALITY")
            print("-" * 40)
            print(f"Intra-cluster Distance: {quality_metrics['avg_intra_cluster_distance']:.4f}")
            print(f"Inter-cluster Distance: {quality_metrics['avg_inter_cluster_distance']:.4f}")
            print(f"Separation Score: {quality_metrics['cluster_separation_score']:.4f}")
            print("(Higher separation score indicates better clustering)")

        print("=" * 60)

    def save_results(self, pred_segments: List[Tuple[float, float, int]],
                     output_file: str, format_type: str = 'json'):
        """
        Save diarization results to file

        Args:
            pred_segments: Predicted segments
            output_file: Output file path
            format_type: 'json', 'rttm', or 'csv'
        """
        if format_type.lower() == 'json':
            results = {
                'segments': [
                    {'start': start, 'end': end, 'speaker': speaker}
                    for start, end, speaker in pred_segments
                ],
                'num_speakers': len(set(seg[2] for seg in pred_segments))
            }
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        elif format_type.lower() == 'rttm':

            filename = os.path.splitext(os.path.basename(output_file))[0]
            with open(output_file, 'w') as f:
                for start, end, speaker in pred_segments:
                    duration = end - start
                    f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <U> <U> speaker_{speaker} <U> <U>\n")

        elif format_type.lower() == 'csv':
            with open(output_file, 'w') as f:
                f.write("start_time,end_time,speaker_id\n")
                for start, end, speaker in pred_segments:
                    f.write(f"{start:.3f},{end:.3f},{speaker}\n")

        print(f"Results saved to {output_file} in {format_type} format")

    def batch_diarize(self, audio_files: List[str], output_dir: str,
                      voice_activity_detection: Dict[str, np.ndarray] = None,
                      num_speakers: Dict[str, int] = None):
        """
        Perform diarization on multiple files

        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save results
            voice_activity_detection: Optional dict mapping filenames to VAD arrays
            num_speakers: Optional dict mapping filenames to number of speakers
        """
        os.makedirs(output_dir, exist_ok=True)

        all_quality_metrics = {}

        for audio_file in audio_files:
            print(f"\nProcessing {audio_file}...")
            filename = os.path.splitext(os.path.basename(audio_file))[0]

            vad = voice_activity_detection.get(filename) if voice_activity_detection else None
            n_spk = num_speakers.get(filename) if num_speakers else None

            pred_labels, pred_segments = self.diarize(audio_file, vad, n_spk)

            quality_metrics = self.analyze_diarization_quality(pred_labels)
            all_quality_metrics[filename] = quality_metrics

            output_file = os.path.join(output_dir, f"{filename}_diarization.json")
            self.save_results(pred_segments, output_file)

            print(f"\nQuality analysis for {filename}:")
            self.print_quality_analysis(quality_metrics)

        summary_file = os.path.join(output_dir, "batch_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(str(all_quality_metrics))

        print(f"\nBatch processing completed. Summary saved to {summary_file}")

    def save_model(self, model_path: str):
        """Save trained models"""
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(self.speaker_models, f)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load trained models"""
        import pickle
        with open(model_path, 'rb') as f:
            self.speaker_models = pickle.load(f)
        print(f"Model loaded from {model_path}")

    def process_and_evaluate_file(args):
        """
        A helper function to be called by each process in the pool.
        It unpacks arguments and calls the evaluation method.
        """
        vbx_system, audio_file_base, true_segments, dataset_path = args

        audio_file_path = None
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(dataset_path, audio_file_base + ext)
            if os.path.exists(path):
                audio_file_path = path
                break

        if audio_file_path is None:
            print(f"Audio file for '{audio_file_base}' not found in dataset.")
            return audio_file_base, None

        print(f"\n--- Evaluating {audio_file_base} ---")
        metrics, _, _ = vbx_system.diarize_and_evaluate(audio_file_path, true_segments)

        return audio_file_base, metrics


if __name__ == "__main__":

    vbx_system = VBxDiarization(frame_rate=100, seg_len=1.5, num_mix=16)

    # vbx_system.train_on_dataset("../diarization_dataset")
    #
    # vbx_system.save_model("vbx_model.pkl")

    vbx_system.load_model("./vbx_model.pkl")

    with open("./ground_truth.json", "r") as f:
        ground_truth_json = json.load(f)

    ground_truth_segments = vbx_system.convert_ground_truth_to_segments(ground_truth_json)

    folder_path = "../test"
    audio_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp3"):
            audio_files.append(os.path.join(folder_path, file_name))

    all_results = {}

    for audio_file in audio_files:

        filename = audio_file.split('/')[-1].replace('.mp3', '')

        gt_segments = None
        for gt_name, segments in ground_truth_segments.items():
            if gt_name in filename or filename in gt_name:
                gt_segments = segments
                break

        if gt_segments is None:
            print(f"No ground truth found for {filename}")
            continue

        print(f"\nEvaluating {filename}...")
        print(f"Ground truth has {len(gt_segments)} segments")

        try:
            metrics, pred_labels, pred_segments = vbx_system.diarize_and_evaluate(
                audio_file, gt_segments, num_speakers=None
            )

            if metrics:
                all_results[filename] = metrics

                vbx_system.print_evaluation_results(metrics)

                print(f"\nKEY METRICS for {filename}:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
                print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
                print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
                print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
                print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
                print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")

        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    if all_results:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)

        accuracies = [r['accuracy'] for r in all_results.values()]
        f1_macros = [r['f1_macro'] for r in all_results.values()]
        precisions = [r['precision_macro'] for r in all_results.values()]
        recalls = [r['recall_macro'] for r in all_results.values()]

        print(f"Average Accuracy: {np.mean(accuracies):.4f}")
        print(f"Average F1-Score (Macro): {np.mean(f1_macros):.4f}")
        print(f"Average Precision (Macro): {np.mean(precisions):.4f}")
        print(f"Average Recall (Macro): {np.mean(recalls):.4f}")

        print("\nIndividual Results:")
        for filename, metrics in all_results.items():
            print(f"{filename}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
