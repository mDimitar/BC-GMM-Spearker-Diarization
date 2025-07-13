import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
import os
import json
import pickle
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class EnhancedIVectorDiarization:
    def __init__(self, frame_rate=100, seg_len=1.5, num_mix=256, ivector_dim=100, reg_covar=1e-4):
        self.frame_rate = frame_rate
        self.seg_len = seg_len
        self.num_mix = num_mix
        self.ivector_dim = ivector_dim
        self.reg_covar = reg_covar
        self.ubm = None
        self.total_variability_matrix = None
        self.within_class_covariance = None
        self.between_class_covariance = None
        self.plda_mean = None
        self.is_trained = False

    def extract_features(self, wav_file: str) -> Optional[np.ndarray]:
        try:
            duration = librosa.get_duration(path=wav_file)
            all_mfccs = []
            chunk_duration = 60.0
            for offset in np.arange(0, duration, chunk_duration):
                wav_data, _ = librosa.load(wav_file, sr=16000, offset=offset, duration=chunk_duration)
                if len(wav_data) == 0: continue
                mfcc = librosa.feature.mfcc(y=wav_data, sr=16000, n_mfcc=40, hop_length=int(16000 / self.frame_rate)).T
                all_mfccs.append(mfcc)
            if not all_mfccs: return None
            return np.vstack(all_mfccs)
        except Exception as e:
            print(f"Error extracting features from {wav_file}: {e}")
            return None

    def train_on_dataset(self, dataset_path: str):
        """Train the entire i-vector system from a dataset."""
        print("Starting i-vector model training...")
        audio_files = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.mp3', '.flac'))]
        if not audio_files:
            print(f"No audio files found in {dataset_path}");
            return

        print("--- Step 1: Collecting features for UBM ---")
        all_features_list = [self.extract_features(os.path.join(dataset_path, f)) for f in audio_files]
        all_features_list = [f for f in all_features_list if f is not None and f.shape[0] > 0]
        if not all_features_list:
            print("ERROR: Could not collect any features for UBM training. Aborting.");
            return

        print("\n--- Step 2: Training UBM ---")
        self.train_ubm(np.vstack(all_features_list))

        print("\n--- Step 3: Computing Baum-Welch statistics ---")
        training_stats = [self.compute_baum_welch_stats(features) for features in all_features_list if
                          features.shape[0] > self.num_mix]
        del all_features_list

        print("\n--- Step 4: Training Total Variability Matrix ---")
        self.train_total_variability_matrix(training_stats)

        print("\n--- Step 5: Extracting i-vectors for PLDA ---")
        ivectors_for_plda, speaker_labels_for_plda = [], []
        for audio_file in audio_files:
            features = self.extract_features(os.path.join(dataset_path, audio_file))
            if features is None or features.shape[0] <= self.num_mix: continue
            speaker_id = os.path.splitext(audio_file)[0]
            ivec = self.extract_ivector(features)
            if ivec is not None:
                ivectors_for_plda.append(ivec)
                speaker_labels_for_plda.append(speaker_id)

        print("\n--- Step 6: Training PLDA Model ---")
        self.train_plda(ivectors_for_plda, speaker_labels_for_plda)

        self.is_trained = True
        print("\n--- i-Vector system training completed! ---")

    def train_ubm(self, all_features: np.ndarray):
        print(f"Training UBM with {self.num_mix} components on {all_features.shape[0]} frames...")
        self.ubm = GaussianMixture(
            n_components=self.num_mix, covariance_type='diag', random_state=42,
            max_iter=20, verbose=1, tol=1e-3, reg_covar=self.reg_covar
        )
        self.ubm.fit(all_features)
        print("UBM training completed.")

    def compute_baum_welch_stats(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.ubm is None: raise ValueError("UBM must be trained first!")
        posteriors = self.ubm.predict_proba(features)
        n_stats = np.sum(posteriors, axis=0)
        f_stats = np.dot(features.T, posteriors).T
        return n_stats, f_stats

    def train_total_variability_matrix(self, training_stats: List[Tuple]):
        print("Training Total Variability Matrix...")
        if not training_stats: print("Warning: No statistics for T-matrix training."); return
        feature_dim = training_stats[0][1].shape[1]
        self.total_variability_matrix = np.random.randn(self.num_mix, feature_dim, self.ivector_dim) * 0.01
        for iteration in range(10):
            print(f"T-matrix EM iteration {iteration + 1}/10")
            E_w, E_ww_T = np.zeros((self.ivector_dim,)), np.zeros((self.ivector_dim, self.ivector_dim))
            A_c, num_samples = np.zeros((self.num_mix, feature_dim, self.ivector_dim)), 0
            for n_stats, f_stats in training_stats:
                w, _ = self._extract_ivector_from_stats(n_stats, f_stats)
                if w is not None:
                    num_samples += 1;
                    E_w += w;
                    E_ww_T += np.outer(w, w)
                    for c in range(self.num_mix): A_c[c] += np.outer(f_stats[c] - n_stats[c] * self.ubm.means_[c], w)
            if num_samples == 0: print("Warning: No valid i-vectors for T-matrix training."); continue
            for c in range(self.num_mix):
                try:
                    self.total_variability_matrix[c] = np.linalg.solve(E_ww_T, A_c[c].T).T
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular matrix in T-matrix update for component {c}."); pass
        print("Total Variability Matrix training completed!")

    def _extract_ivector_from_stats(self, n_stats: np.ndarray, f_stats: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            L_c = np.eye(self.ivector_dim)
            linear_term = np.zeros((self.ivector_dim,))
            for c in range(self.num_mix):
                T_c = self.total_variability_matrix[c]
                L_c += n_stats[c] * np.dot(T_c.T, T_c)
                linear_term += np.dot(T_c.T, f_stats[c] - n_stats[c] * self.ubm.means_[c])
            covariance = np.linalg.inv(L_c)
            return np.dot(covariance, linear_term), covariance
        except np.linalg.LinAlgError:
            return None, None

    def extract_ivector(self, features: np.ndarray) -> Optional[np.ndarray]:
        if self.ubm is None or self.total_variability_matrix is None: raise ValueError("Model must be trained first!")
        if features.shape[0] < self.num_mix: return None
        n_stats, f_stats = self.compute_baum_welch_stats(features)
        ivector, _ = self._extract_ivector_from_stats(n_stats, f_stats)
        return ivector

    def train_plda(self, ivectors: List[np.ndarray], speaker_labels: List[str]):
        print("Training PLDA model...")
        valid_data = [(iv, lbl) for iv, lbl in zip(ivectors, speaker_labels) if iv is not None]
        if not valid_data: print("Warning: No valid i-vectors for PLDA training!"); return
        ivectors, speaker_labels = zip(*valid_data)
        ivectors = np.array(ivectors);
        unique_speakers = list(set(speaker_labels))
        self.plda_mean = np.mean(ivectors, axis=0)
        centered_ivectors = ivectors - self.plda_mean
        ivector_dim = ivectors.shape[1]
        within_scatter, between_scatter = np.zeros((ivector_dim, ivector_dim)), np.zeros((ivector_dim, ivector_dim))
        for speaker in unique_speakers:
            speaker_ivectors = centered_ivectors[np.array(speaker_labels) == speaker]
            if speaker_ivectors.shape[0] == 0: continue
            speaker_mean = np.mean(speaker_ivectors, axis=0)
            within_scatter += np.dot((speaker_ivectors - speaker_mean).T, (speaker_ivectors - speaker_mean))
            between_scatter += len(speaker_ivectors) * np.outer(speaker_mean, speaker_mean)
        self.within_class_covariance = within_scatter / len(ivectors)
        self.between_class_covariance = between_scatter / len(unique_speakers)
        print("PLDA training completed!")

    def plda_score(self, ivector1: np.ndarray, ivector2: np.ndarray) -> float:
        if self.within_class_covariance is None or self.between_class_covariance is None or self.plda_mean is None:
            return 1 - cosine(ivector1, ivector2)
        try:
            iv1_centered, iv2_centered = ivector1 - self.plda_mean, ivector2 - self.plda_mean
            S_w_inv = np.linalg.pinv(self.within_class_covariance)
            S_b_inv = np.linalg.pinv(self.between_class_covariance)
            score = np.dot(iv1_centered.T, S_b_inv @ iv1_centered) + \
                    np.dot(iv2_centered.T, S_b_inv @ iv2_centered) - \
                    np.dot((iv1_centered - iv2_centered).T, S_w_inv @ (iv1_centered - iv2_centered))
            return float(score)
        except np.linalg.LinAlgError as e:
            print(f"!!! WARNING: PLDA scoring failed. Falling back to cosine distance. Error: {e} !!!")
            return 1 - cosine(ivector1, ivector2)

    def hierarchical_clustering(self, ivectors: List[np.ndarray], num_speakers: int = None,
                                threshold: float = 0.0) -> np.ndarray:
        valid_ivectors = [iv for iv in ivectors if iv is not None]
        if len(valid_ivectors) < 2: return np.array([0] * len(ivectors))
        n_segments = len(valid_ivectors)
        distance_matrix = np.zeros(int(n_segments * (n_segments - 1) / 2))
        k = 0
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                distance_matrix[k] = -self.plda_score(valid_ivectors[i], valid_ivectors[j]);
                k += 1
        linkage_matrix = linkage(distance_matrix, method='average')
        if num_speakers is None:
            valid_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
        else:
            valid_labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')
        valid_labels -= 1
        full_labels = np.full(len(ivectors), -1, dtype=int)
        valid_indices = [i for i, iv in enumerate(ivectors) if iv is not None]
        for i, orig_idx in enumerate(valid_indices): full_labels[orig_idx] = valid_labels[i]
        last_valid_label = 0
        for i in range(len(full_labels)):
            if full_labels[i] != -1:
                last_valid_label = full_labels[i]
            else:
                full_labels[i] = last_valid_label
        return full_labels

    def diarize(self, wav_file: str, num_speakers: int = None) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        if not self.is_trained: raise ValueError("Model must be trained first!")
        mfcc_features = self.extract_features(wav_file)
        if mfcc_features is None: return np.array([]), []
        seg_size = int(self.frame_rate * self.seg_len)
        ivectors = [self.extract_ivector(mfcc_features[start:min(start + seg_size, mfcc_features.shape[0])])
                    for start in range(0, mfcc_features.shape[0], seg_size)]
        speaker_labels = self.hierarchical_clustering(ivectors, num_speakers=num_speakers)
        segments = [(i * self.seg_len, (i + 1) * self.seg_len, int(sid)) for i, sid in enumerate(speaker_labels)]
        return speaker_labels, segments

    def diarize_and_evaluate(self, wav_file: str, true_segments: List[Tuple[float, float, int]]):
        num_true_speakers = len(set(seg[2] for seg in true_segments)) if true_segments else 1
        pred_labels, _ = self.diarize(wav_file, num_speakers=num_true_speakers)
        audio_duration = librosa.get_duration(path=wav_file)
        num_segments = int(np.ceil(audio_duration / self.seg_len))
        true_labels = np.zeros(num_segments, dtype=int)
        for start_time, end_time, speaker_id in true_segments:
            start_seg = int(start_time / self.seg_len);
            end_seg = int(end_time / self.seg_len)
            true_labels[max(0, start_seg):min(num_segments, end_seg)] = speaker_id
        metrics, aligned_pred_labels = self.full_evaluate_diarization(true_labels, pred_labels)
        min_len = min(len(true_labels), len(aligned_pred_labels))
        return metrics, true_labels[:min_len], aligned_pred_labels[:min_len]

    def save_model(self, model_path: str):
        """Saves the trained model state to a file."""
        model_state = {
            'frame_rate': self.frame_rate, 'seg_len': self.seg_len, 'num_mix': self.num_mix,
            'ivector_dim': self.ivector_dim, 'reg_covar': self.reg_covar, 'ubm': self.ubm,
            'total_variability_matrix': self.total_variability_matrix,
            'within_class_covariance': self.within_class_covariance,
            'between_class_covariance': self.between_class_covariance,
            'plda_mean': self.plda_mean, 'is_trained': self.is_trained
        }
        with open(model_path, 'wb') as f: pickle.dump(model_state, f)
        print(f"Model state saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str):
        """Loads a trained model from a file by restoring its state."""
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        instance = cls(
            frame_rate=model_state.get('frame_rate', 100), seg_len=model_state.get('seg_len', 1.5),
            num_mix=model_state.get('num_mix', 256), ivector_dim=model_state.get('ivector_dim', 100),
            reg_covar=model_state.get('reg_covar', 1e-4)
        )
        instance.ubm = model_state.get('ubm')
        instance.total_variability_matrix = model_state.get('total_variability_matrix')
        instance.within_class_covariance = model_state.get('within_class_covariance')
        instance.between_class_covariance = model_state.get('between_class_covariance')
        instance.plda_mean = model_state.get('plda_mean')
        instance.is_trained = model_state.get('is_trained', False)
        print(f"Model loaded from {model_path}")
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
    model_file = os.path.abspath('ivector_model.pkl')

    # # --- TRAINING ---
    # if not os.path.exists(model_file):
    #     print(f"Model file not found at {model_file}. Starting training...")
    #     dataset_folder = './pythonProject/dataset'
    #     if not os.path.isdir(dataset_folder):
    #         raise FileNotFoundError(f"Training dataset folder not found at: {dataset_folder}")
    #     ivector_system = EnhancedIVectorDiarization()
    #     ivector_system.train_on_dataset(dataset_folder)
    #     ivector_system.save_model(model_file)
    #     print("\n--- Model training complete. Now starting evaluation. ---")

    # --- EVALUATION ---
    print("\n--- Starting Sequential Evaluation ---")
    eval_system = EnhancedIVectorDiarization.load_model(model_file)

    with open("./ground_truth.json", "r") as f:
        ground_truth_json = json.load(f)

    true_segments_data = eval_system.convert_ground_truth_to_segments(ground_truth_json)
    dataset_path = "./pythonProject/evaluation_data"
    all_true_labels, all_pred_labels = [], []

    for audio_file_base, gt_segments in true_segments_data.items():
        audio_file_path = None
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(dataset_path, audio_file_base + ext)
            if os.path.exists(path): audio_file_path = path; break
        if not audio_file_path:
            print(f"\n--- Evaluation failed for {audio_file_base}: File not found ---");
            continue

        print(f"\n--- Evaluating {audio_file_base} ---")
        metrics, true_labels, pred_labels = eval_system.diarize_and_evaluate(
            wav_file=audio_file_path, true_segments=gt_segments
        )
        if metrics:
            eval_system.print_full_evaluation_results(metrics)
            if true_labels is not None and pred_labels is not None:
                all_true_labels.extend(true_labels);
                all_pred_labels.extend(pred_labels)

    if all_true_labels and all_pred_labels:
        print("\n\n" + "#" * 70 + "\n### OVERALL COMBINED I-VECTOR METRICS FOR THE ENTIRE DATASET\n" + "#" * 70)
        final_true, final_pred = np.array(all_true_labels), np.array(all_pred_labels)
        overall_metrics, _ = eval_system.full_evaluate_diarization(final_true, final_pred)
        eval_system.print_full_evaluation_results(overall_metrics)

    print("\n--- i-vector Evaluation Script Completed ---")
