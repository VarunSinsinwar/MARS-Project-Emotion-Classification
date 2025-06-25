import numpy as np
import librosa
import tensorflow as tf
import sys

model = tf.keras.models.load_model('emotion_classifier_gru.h5')

emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_features(file_path, max_len=200):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs.T

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)[0]
    emotion = emotion_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_emotion.py path_to_audio_file")
        sys.exit(1)

    audio_file = sys.argv[1]
    emotion, confidence = predict_emotion(audio_file)
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
