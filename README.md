# Speaker Diarization Script

This Python script performs **speaker diarization** on an input audio file. It uses a pre-trained binary classifier to predict the number of speakers and then applies Gaussian Mixture Models (GMMs), voice activity detection (VAD), and agglomerative clustering to segment the audio into speaker-labeled sections.

## What It Does

1. Loads a binary classifier model from `../binary_classifier/best_model.pkl`.
2. Loads and processes the test audio file.
3. Predicts the number of speakers.
4. Applies segmentation and clustering based on the predicted speaker count.
5. Prints a DataFrame showing diarization results with time segments and speaker labels.

## Requirements

## Important notice: Make sure you run your IDE as administrator

This is important due to some models are required write permissions in directories for saving
the trained model.

Ensure you have Python 3.7+ and the following Python packages installed:

- `numpy`
- `librosa`
- `scikit-learn`
- `pickle` (built-in with Python)
- Any custom functions: `process_audio`, `voice_activity_detection`, `train_gmm`, `segment_frame`, and `speakerdiarisationdf`

Install the required packages using pip:

```bash
pip install numpy librosa scikit-learn
```
## HOW TO: Use the speaker count predictor

- Navigate in `number_of_speakers_predictor` folder
- Change the path to the needed audio file in the `input_audio_file` variable
- Execute `python predict_number_of_speakers.py`

## HOW TO: Use the diarization module

- Navigate in `diarizer` folder
- Change the path to the needed audio file in the `input_audio_file` variable
- Execute `python diarization.py`

## HOW TO: Evaluate the performance of the competing models

- Navigate in `evaluations_of_competing_models` folder
- Open the wanted model script,for ex. `VBx.py` python file
- Execute `python <script-name>.py`, for ex `python VBx.py`