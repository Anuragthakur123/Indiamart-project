# Speech Emotion Recognition

Multi-class classification of emotion from raw speech audio, with a live microphone-based Streamlit demo on top. Given a 3-second voice recording, the trained model predicts one of 8 emotions: Surprise, Neutral, Calm, Happy, Sad, Angry, Fearful, or Disgust.

## Why it's interesting

The notebook isn't a single model trained once — it's a deliberate, broad comparison across very different approaches to the same problem: a CNN-LSTM (Bidirectional), a simpler LSTM, an "improved" attention+Bidirectional variant, and a set of classical ML baselines (SVM, Gradient Boosting, Random Forest, plus unsupervised Bayesian GMM and HMM). The results aren't uniformly flattering — the "improved" model actually underperforms the simpler LSTM it was meant to improve on, and every classical ML baseline is far behind the deep learning models. Keeping that outcome rather than only reporting the best number is closer to how model selection actually works in practice.

The best-performing model is deployed in a small Streamlit app (`src/app.py`) that records audio from the microphone and classifies it live — the app reimplements the exact same MFCC feature-extraction pipeline used during training (`feature_mfcc`/`get_waveforms`/`preprocess_audio`), so predictions are consistent with what the model actually saw at training time. This was checked explicitly while rebuilding this repo: the app's hardcoded emotion label order was cross-verified against the notebook's actual training label encoding (`emotions_dict`) to rule out a silent label-order mismatch — they match.

## Tech stack

Python · librosa (MFCC feature extraction) · TensorFlow/Keras (CNN-LSTM, LSTM) · scikit-learn, hmmlearn (classical ML baselines) · Streamlit + sounddevice/soundfile (live microphone demo)

## Dataset

[RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (not included in this repo — ~450MB). 24 actors (12 male, 12 female) × 60 audio files each = 1,440 `.wav` files across 8 emotion classes. To reproduce training, download the dataset and update the hardcoded `parent_folder` path near the top of the notebook to point at your local copy.

## Results

Test-set accuracy across every architecture tried, in the order they appear in the notebook:

| Model | Test accuracy | Notes |
|---|---|---|
| **CNN-LSTM (Bidirectional)** | **93.46%** | Best model — this is the one saved as `model.h5` and used in the app |
| Simple LSTM | 86.57% | Full per-class precision/recall/F1 in the notebook |
| "Improved" LSTM (+ Attention, + Bidirectional) | 74.53% | Despite the name, underperforms the simpler LSTM above — kept as an honest result, not discarded |
| Classical ML (SVM / Gradient Boosting / Random Forest / GMM / HMM) | ~30–40% | Included for comparison; confirms deep learning's clear advantage on raw MFCC features for this task |

The notebook itself is explicitly framed as an experimental comparison rather than a single polished pipeline — see its intro note for context.

## How to run the app

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Click **Record** to capture 3 seconds of audio, then **Predict** to classify it. The app loads `model/model.h5` and the image assets in `app_assets/` using paths relative to the project root, so it works regardless of which directory you launch it from.

## Project structure

```
src/
  app.py                        Streamlit live-demo app
model/
  model.h5                      Trained CNN-LSTM (Bidirectional) model, 93.46% test accuracy
notebooks/
  SpeechEmotionRecognition.ipynb  Full training/comparison notebook, plus its figures
app_assets/                     Images used by the Streamlit app
```

## License

MIT — see [LICENSE](LICENSE).
