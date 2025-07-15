import numpy as np
import librosa
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
import os
import json
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
import gc

warnings.filterwarnings('ignore')


class EnhancedXVectorDiarization:
    def __init__(self, seg_len=1.5, xvector_model_source="speechbrain/spkrec-ecapa-voxceleb", batch_size=32):
        self.seg_len = seg_len
        self.xvector_model_source = xvector_model_source
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Core components
        self.xvector_extractor = None
        self.within_class_covariance = None
        self.between_class_covariance = None
        self.plda_mean = None
        self.is_trained = False
        self._load_xvector_model()

    def _load_xvector_model(self):
        """Loads the pretrained x-vector model from SpeechBrain."""
        print(f"Loading x-vector model: {self.xvector_model_source}...")
        try:
            self.xvector_extractor = EncoderClassifier.from_hparams(
                source=self.xvector_model_source,
                savedir=f"pretrained_models/{self.xvector_model_source.split('/')[-1]}",
                run_opts={"device": self.device}
            )
            print("X-vector model loaded successfully.")
        except Exception as e:
            print(f"Error loading x-vector model: {e}")
            raise

    def _load_audio(self, wav_file: str) -> Optional[torch.Tensor]:
        """Loads and preprocesses an audio file."""
        try:
            wav_data, sr = torchaudio.load(wav_file)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav_data = resampler(wav_data)
            if wav_data.shape[0] > 1:
                wav_data = torch.mean(wav_data, dim=0, keepdim=True)
            return wav_data
        except Exception as e:
            print(f"Error loading audio file {wav_file}: {e}")
            return None

    def extract_xvectors(self, waveform: torch.Tensor) -> List[Optional[np.ndarray]]:
        """Extracts x-vectors from a waveform by segmenting it."""
        seg_len_samples = int(self.seg_len * 16000)
        segments = [waveform[:, s:min(s + seg_len_samples, waveform.shape[1])] for s in
                    range(0, waveform.shape[1], seg_len_samples)]

        valid_segments, valid_indices = [], []
        for i, segment in enumerate(segments):
            if segment is not None and segment.shape[1] >= 4000:  # Min length for ECAPA
                valid_segments.append(segment.squeeze(0))
                valid_indices.append(i)

        if not valid_segments:
            return [None] * len(segments)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(valid_segments), self.batch_size):
                batch = valid_segments[i:i + self.batch_size]
                max_len = max(seg.shape[0] for seg in batch)
                padded_batch = [torch.nn.functional.pad(seg, (0, max_len - seg.shape[0])) for seg in batch]
                batch_tensor = torch.stack(padded_batch).to(self.device)
                embeddings = self.xvector_extractor.encode_batch(batch_tensor).squeeze(1)
                all_embeddings.extend([emb.cpu().numpy() for emb in embeddings])

        result_embeddings = [None] * len(segments)
        for i, orig_idx in enumerate(valid_indices):
            result_embeddings[orig_idx] = all_embeddings[i]

        return result_embeddings

    def train_on_dataset(self, dataset_path: str):
        """Trains the PLDA model sequentially on x-vectors extracted from the dataset."""
        print("Starting sequential x-vector + PLDA model training...")
        audio_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
                       f.endswith(('.wav', '.mp3', '.flac'))]
        if not audio_files:
            print(f"No audio files found in {dataset_path}");
            return

        all_xvectors, all_labels = [], []
        for i, file_path in enumerate(audio_files):
            print(f"  -> Processing file {i + 1}/{len(audio_files)}: {os.path.basename(file_path)}")
            waveform = self._load_audio(file_path)
            if waveform is None: continue

            xvectors_from_file = self.extract_xvectors(waveform)
            speaker_id = os.path.splitext(os.path.basename(file_path))[0]

            for xv in xvectors_from_file:
                if xv is not None:
                    all_xvectors.append(xv)
                    all_labels.append(speaker_id)
            del waveform, xvectors_from_file
            gc.collect()

        if not all_xvectors:
            print("ERROR: Could not extract any x-vectors for training. Aborting.");
            return

        print(f"\nTraining PLDA on {len(all_xvectors)} x-vectors from {len(audio_files)} files...")
        self.train_plda(all_xvectors, all_labels)
        self.is_trained = True
        print("\n--- x-vector + PLDA system training completed! ---")

    def train_plda(self, xvectors: List[np.ndarray], speaker_labels: List[str]):
        print("Training PLDA model...")
        xvectors = np.array(xvectors)
        unique_speakers = list(set(speaker_labels))
        self.plda_mean = np.mean(xvectors, axis=0)
        centered_xvectors = xvectors - self.plda_mean

        xvector_dim = xvectors.shape[1]
        within_scatter = np.zeros((xvector_dim, xvector_dim))
        between_scatter = np.zeros((xvector_dim, xvector_dim))

        for speaker in unique_speakers:
            speaker_xvectors = centered_xvectors[np.array(speaker_labels) == speaker]
            if speaker_xvectors.shape[0] == 0: continue
            speaker_mean = np.mean(speaker_xvectors, axis=0)
            within_scatter += np.dot((speaker_xvectors - speaker_mean).T, (speaker_xvectors - speaker_mean))
            between_scatter += len(speaker_xvectors) * np.outer(speaker_mean, speaker_mean)

        self.within_class_covariance = within_scatter / len(xvectors)
        self.between_class_covariance = between_scatter / len(unique_speakers)
        print("PLDA training completed!")

    def plda_score(self, xvector1: np.ndarray, xvector2: np.ndarray) -> float:
        if self.within_class_covariance is None or self.between_class_covariance is None or self.plda_mean is None:
            return 1 - cosine(xvector1, xvector2)
        try:
            xv1_centered, xv2_centered = xvector1 - self.plda_mean, xvector2 - self.plda_mean
            S_w_inv = np.linalg.pinv(self.within_class_covariance)
            S_b_inv = np.linalg.pinv(self.between_class_covariance)
            score = np.dot(xv1_centered.T, S_b_inv @ xv1_centered) + \
                    np.dot(xv2_centered.T, S_b_inv @ xv2_centered) - \
                    np.dot((xv1_centered - xv2_centered).T, S_w_inv @ (xv1_centered - xv2_centered))
            return float(score)
        except np.linalg.LinAlgError:
            return 1 - cosine(xvector1, xvector2)

    def hierarchical_clustering(self, xvectors: List[np.ndarray], num_speakers: int = None,
                                threshold: float = 0.0) -> np.ndarray:
        valid_xvectors = [xv for xv in xvectors if xv is not None]
        if len(valid_xvectors) < 2: return np.array([0] * len(xvectors))

        n_segments = len(valid_xvectors)

        # Create a square similarity matrix
        score_matrix = np.zeros((n_segments, n_segments))
        for i in range(n_segments):
            for j in range(i, n_segments):
                score = self.plda_score(valid_xvectors[i], valid_xvectors[j])
                score_matrix[i, j] = score
                score_matrix[j, i] = score

        # Convert similarity to distance: distance = max_score - score
        # This ensures all distances are non-negative.
        distance_matrix = np.max(score_matrix) - score_matrix

        # Get the condensed distance matrix (upper triangle) for the linkage function
        condensed_distance = distance_matrix[np.triu_indices(n_segments, k=1)]

        linkage_matrix = linkage(condensed_distance, method='average')

        if num_speakers is None:
            # The threshold for distance-based clustering needs to be tuned.
            # We will stick to the num_speakers case for evaluation.
            labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
        else:
            labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')

        labels -= 1
        full_labels = np.full(len(xvectors), -1, dtype=int)
        valid_indices = [i for i, xv in enumerate(xvectors) if xv is not None]
        for i, orig_idx in enumerate(valid_indices): full_labels[orig_idx] = labels[i]

        last_valid_label = 0
        for i in range(len(full_labels)):
            if full_labels[i] == -1:
                full_labels[i] = last_valid_label
            else:
                last_valid_label = full_labels[i]

        return full_labels

    def diarize(self, wav_file: str, num_speakers: int = None) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        if not self.is_trained: raise ValueError("PLDA model must be trained first!")
        waveform = self._load_audio(wav_file)
        if waveform is None: return np.array([]), []
        xvectors = self.extract_xvectors(waveform)
        speaker_labels = self.hierarchical_clustering(xvectors, num_speakers=num_speakers)
        segments = [(i * self.seg_len, (i + 1) * self.seg_len, int(sid)) for i, sid in enumerate(speaker_labels)]
        return speaker_labels, segments

    def diarize_and_evaluate(self, wav_file: str, true_segments: List[Tuple[float, float, int]]):
        num_true_speakers = len(set(seg[2] for seg in true_segments)) if true_segments else 1
        pred_labels, _ = self.diarize(wav_file, num_speakers=num_true_speakers)

        audio_duration = librosa.get_duration(path=wav_file)
        num_segments = int(np.ceil(audio_duration / self.seg_len))
        true_labels = np.zeros(num_segments, dtype=int)
        for start_time, end_time, speaker_id in true_segments:
            start_seg, end_seg = int(start_time / self.seg_len), int(end_time / self.seg_len)
            true_labels[max(0, start_seg):min(num_segments, end_seg)] = speaker_id

        metrics, _ = self.full_evaluate_diarization(true_labels, pred_labels)
        min_len = min(len(true_labels), len(pred_labels))
        return metrics, true_labels[:min_len], pred_labels[:min_len]

    def save_model(self, model_path: str):
        model_state = {
            'plda_mean': self.plda_mean,
            'within_class_covariance': self.within_class_covariance,
            'between_class_covariance': self.between_class_covariance,
            'is_trained': self.is_trained,
            'seg_len': self.seg_len,
            'xvector_model_source': self.xvector_model_source,
            'batch_size': self.batch_size
        }
        with open(model_path, 'wb') as f: pickle.dump(model_state, f)
        print(f"PLDA model state saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str):
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        instance = cls(
            seg_len=model_state.get('seg_len', 1.5),
            xvector_model_source=model_state.get('xvector_model_source', "speechbrain/spkrec-ecapa-voxceleb"),
            batch_size=model_state.get('batch_size', 32)
        )
        instance.plda_mean = model_state.get('plda_mean')
        instance.within_class_covariance = model_state.get('within_class_covariance')
        instance.between_class_covariance = model_state.get('between_class_covariance')
        instance.is_trained = model_state.get('is_trained', False)
        print(f"PLDA model loaded from {model_path}")
        return instance

    @staticmethod
    def parse_time_to_seconds(time_str: str) -> float:
        time_str = time_str.strip().replace('s', '')
        if ':' in time_str:
            parts = time_str.split(':');
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)

    def convert_ground_truth_to_segments(self, ground_truth_json: List[Dict]) -> Dict[
        str, List[Tuple[float, float, int]]]:
        converted_data = {}
        for item in ground_truth_json:
            audio_file_full = item["audio_file"];
            audio_file_base, _ = os.path.splitext(audio_file_full)
            segments = []
            for speaker_name, time_ranges in item["diarization"]["speakers"][0].items():
                speaker_id = int(speaker_name.split('_')[1]) - 1
                for time_range in time_ranges:
                    try:
                        start_str, end_str = time_range.split(' - ')
                        segments.append(
                            (self.parse_time_to_seconds(start_str), self.parse_time_to_seconds(end_str), speaker_id))
                    except ValueError:
                        continue
            converted_data[audio_file_base] = sorted(segments, key=lambda x: x[0])
        return converted_data

    def _align_labels(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
        true_unique, pred_unique = np.unique(true_labels), np.unique(pred_labels)
        if len(pred_unique) == 0: return np.array([])
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=np.union1d(true_unique, pred_unique))
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        label_mapping = {pred_unique[p_idx]: true_unique[t_idx] for p_idx, t_idx in zip(col_ind, row_ind) if
                         p_idx < len(pred_unique) and t_idx < len(true_unique)}
        max_true_label = max(true_unique) if len(true_unique) > 0 else -1
        return np.array([label_mapping.get(label, max_true_label + 1 + label) for label in pred_labels])

    def full_evaluate_diarization(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[Dict, np.ndarray]:
        min_len = min(len(true_labels), len(pred_labels))
        true_labels, pred_labels = true_labels[:min_len], pred_labels[:min_len]
        if min_len == 0: return {}, np.array([])
        aligned_pred_labels = self._align_labels(true_labels, pred_labels)
        metrics = {'accuracy': accuracy_score(true_labels, aligned_pred_labels),
                   'precision_macro': precision_score(true_labels, aligned_pred_labels, average='macro',
                                                      zero_division=0),
                   'recall_macro': recall_score(true_labels, aligned_pred_labels, average='macro', zero_division=0),
                   'f1_macro': f1_score(true_labels, aligned_pred_labels, average='macro', zero_division=0),
                   'num_true_speakers': len(np.unique(true_labels)),
                   'num_pred_speakers': len(np.unique(aligned_pred_labels)),
                   'confusion_matrix': confusion_matrix(true_labels, aligned_pred_labels).tolist()}
        return metrics, aligned_pred_labels

    def print_full_evaluation_results(self, metrics: Dict):
        print("\n" + "=" * 60 + "\nDIARIZATION EVALUATION RESULTS\n" + "=" * 60)
        print(f"Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"Number of True Speakers:  {metrics.get('num_true_speakers', 0)}")
        print(f"Number of Pred Speakers:  {metrics.get('num_pred_speakers', 0)}")
        print("\n" + "-" * 40 + "\nMACRO AVERAGED METRICS\n" + "-" * 40)
        print(f"Precision (Macro):  {metrics.get('precision_macro', 0):.4f}")
        print(f"Recall (Macro):     {metrics.get('recall_macro', 0):.4f}")
        print(f"F1-Score (Macro):   {metrics.get('f1_macro', 0):.4f}")
        if 'confusion_matrix' in metrics:
            print("\n" + "-" * 40 + "\nCONFUSION MATRIX\n" + "-" * 40)
            print(np.array(metrics['confusion_matrix']))
        print("=" * 60)


if __name__ == "__main__":

    model_file = './xvector_plda_model.pkl'
    #
    # # --- TRAINING ---
    # if not os.path.exists(model_file):
    #     print(f"Model file not found at {model_file}. Starting training...")
    #     dataset_folder = "../diarization_dataset" #local path to our dataset
    #     if not os.path.isdir(dataset_folder):
    #         raise FileNotFoundError(f"Training dataset folder not found at: {dataset_folder}")
    #
    #     xvector_system = EnhancedXVectorDiarization()
    #     xvector_system.train_on_dataset(dataset_folder)
    #
    #     os.makedirs(os.path.dirname(model_file), exist_ok=True)
    #     xvector_system.save_model(model_file)
    #     print("\n--- Model training complete. Now starting evaluation. ---")

    # --- EVALUATION ---
    print("\n--- Starting Parallel x-vector + PLDA Evaluation ---")
    main_eval_system = EnhancedXVectorDiarization()

    with open("./ground_truth.json", "r") as f:
        ground_truth_json = json.load(f)

    eval_system = EnhancedXVectorDiarization.load_model(model_file)
    true_segments_data = eval_system.convert_ground_truth_to_segments(ground_truth_json)
    dataset_path = "../test"

    all_true_labels, all_pred_labels = [], []

    for audio_file_base, gt_segments in true_segments_data.items():
        audio_file_path = None
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(dataset_path, audio_file_base + ext)
            if os.path.exists(path):
                audio_file_path = path
                break
        if not audio_file_path:
            print(f"\n--- Evaluation failed for {audio_file_base}: File not found ---")
            continue

        print(f"\n--- Evaluating {audio_file_base} ---")
        metrics, true_labels, pred_labels = eval_system.diarize_and_evaluate(
            wav_file=audio_file_path, true_segments=gt_segments
        )
        if metrics:
            eval_system.print_full_evaluation_results(metrics)
            if true_labels is not None and pred_labels is not None:
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)

    if all_true_labels and all_pred_labels:
        print("\n\n" + "#" * 70 + "\n### OVERALL COMBINED X-VECTOR METRICS FOR THE ENTIRE DATASET\n" + "#" * 70)
        final_true, final_pred = np.array(all_true_labels), np.array(all_pred_labels)
        overall_metrics, _ = eval_system.full_evaluate_diarization(final_true, final_pred)
        eval_system.print_full_evaluation_results(overall_metrics)

    print("\n--- x-vector + PLDA Evaluation Script Completed ---")

