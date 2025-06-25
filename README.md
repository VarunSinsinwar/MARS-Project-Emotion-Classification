# MARS-Project-Emotion-Classification
This project is a deep learning based application that classifies human emotions from raw speech audio files. It leverages a deep learning pipeline combining Convolutional Neural Networks (CNN), Gated Recurrent Units (GRU), and an Attention mechanism for accurate and context-aware emotion recognition.
## Data Preprocessing
RAVDESS dataset (provided by MARS) was used for training and testing. All the .wav files were used to extract Mel-Frequency Cepstral Coefficients (MFCC) features. 40 MFCCs were extracted for each time step. Padding was done after which each file had 200 time steps (frames). Data augmentation was used, since the data was small (around 2500 .wav files). 4 times data augmentation was done using time stretching, pitch shift, and noise addition. Librosa library was used for all the audio file related preprocessing.
## Model Building
In the model there are two layers of CNN (conv 1D) and then 1 GRU layer. Finally an attention layer is added connected to a fully connected (dense) layer which gives Emotion as the output. Emotions were neutral, calm, happy, sad, angry, fearful, disgust and surprised. Following graph shows the change of accuracies with the epochs.

![download](https://github.com/user-attachments/assets/2017311a-3d70-4746-8a7d-dd85aac749a5)

We ran 100 epochs and result saturated at about 55th epoch. An overall accuracy of 83% was achieved with really good F1 scores of each class. Model has shown low results in some classes (Neutral and sad primarily) which can be improves by increasing the size of the data. Following is the classification report and confusion matrix after the evaluation on test split.
  precision    recall  f1-score   support

       angry       0.95      0.92      0.93        75
        calm       0.87      0.89      0.88        75
     disgust       0.87      0.87      0.87        39
     fearful       0.74      0.81      0.78        75
       happy       0.92      0.80      0.86        76
     neutral       0.70      0.82      0.76        38
         sad       0.75      0.73      0.74        75
    surprised       0.82      0.79      0.81        39

    accuracy                           0.83       492
    macro avg       0.83      0.83      0.83       492
    weighted avg       0.84      0.83      0.83       492
![download](https://github.com/user-attachments/assets/72ecd2f1-6fac-4083-8632-35e822f6b157)

## App building
A web application was built using streamlit and it was deployed on streamlit cloud. The link is https://emotion-classifier-app-de8vbayzsxqucjylat9bpm.streamlit.app/
This app can be used to upload an audio file and then using the trained model, it can predict the emotion in the audio file. Preview video is being added.

## Conclusion
This project is a great example of audio processing and using them for training deep learning models. It can be used for further improvements in AI field and can be directly deployed publicly.
### Libraries used
TensorFlow, Librosa, Streamlit, Pandas, numpy, matplotlib
### Applications used
Google Colab, VS Code, StreamlitCloud
