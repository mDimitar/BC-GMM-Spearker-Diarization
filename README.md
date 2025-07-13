# Speaker Diarization Script

This Python script performs **speaker diarization** on an input audio file. It uses a pre-trained binary classifier to predict the number of speakers and then applies Gaussian Mixture Models (GMMs), voice activity detection (VAD), and agglomerative clustering to segment the audio into speaker-labeled sections.

## What It Does

1. Loads a binary classifier model from `../binary_classifier/best_model.pkl`.
2. Loads and processes the test audio file.
3. Predicts the number of speakers.
4. Applies segmentation and clustering based on the predicted speaker count.
5. Prints a DataFrame showing diarization results with time segments and speaker labels.

## Requirements

Ensure you have Python 3.7+ and the following Python packages installed:

- `numpy`
- `librosa`
- `scikit-learn`
- `pickle` (built-in with Python)
- Any custom functions: `process_audio`, `voice_activity_detection`, `train_gmm`, `segment_frame`, and `speakerdiarisationdf`

Install the required packages using pip:

```bash
pip install numpy librosa scikit-learn
