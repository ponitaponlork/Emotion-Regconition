# Emotion-Regconition

🎭 Facial Emotion Recognition (CNN)

A CNN-based facial emotion recognition system that detects human emotions from facial images captured by a camera.
Trained on the FER-2013 dataset, the model classifies six emotions:

Angry · Fear · Happy · Sad · Surprise · Neutral

🔍 Overview

This project uses a Convolutional Neural Network (CNN) trained from scratch to recognize facial expressions from 48×48 grayscale images.
Data preprocessing, augmentation, and class balancing were applied to improve performance and reduce overfitting.

📂 Dataset

Source: FER-2013

Image size: 48×48 (grayscale)

Classes used: 6 (Disgust removed due to imbalance)

Split: 70% train / 15% validation / 15% test

🧠 Model

CNN with 3 convolution blocks

Batch Normalization + MaxPooling

Dropout (0.25–0.5)

Dense layer (1024 units)

Softmax output (6 classes)

Optimizer: Adam
Loss: Categorical Cross-Entropy

🛠️ Data Augmentation

Rotation (±10°)

Zoom, shear

Width & height shift

Horizontal flip

📊 Results

Accuracy: 59%

Precision: 0.59

Recall: 0.59

F1-score: 0.58

Best performance on Happy and Surprise, weaker on Fear and Sad.

📁 Outputs

emotion_model_best.h5 – trained model

class_labels.json – emotion mapping

Evaluation metrics & confusion matrix
