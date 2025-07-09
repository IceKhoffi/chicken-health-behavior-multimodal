import torch
import numpy as np
import librosa
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F

class ModdifiedModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ModdifiedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Function to convert audio to Log-Mel Spectrogram
def audio_to_log_mel_spec(filepath, sr=22050, wav_size=int(1.5 * 22050)):
    wav, sr = librosa.load(filepath, sr=sr)
    wav = librosa.util.normalize(wav)

    if len(wav) > wav_size:
        wav = wav[:wav_size]
    else:
        wav = np.pad(wav, (0, max(0, wav_size - len(wav))), "constant")

    n_fft = 2048
    hop_length = 256
    n_mels = 128

    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return wav, mel_spec_db

# Load trained model
def load_model(model_path):
    model = ModdifiedModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Predict function
def vocaliaztion_prediction(audio_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, X = audio_to_log_mel_spec(audio_path)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    X_tensor = X_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(X_tensor)

    pred_class = torch.argmax(output, dim=1).item()
    probabilities = F.softmax(output, dim=1)[0]

    class_labels = {0: 'Healthy', 1: 'Noise', 2: 'Unhealthy'}
    print(f"Predicted class: {class_labels[pred_class]}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_labels[i]}: {prob.item() * 100:.2f}%")

def main(args):
    audio_path = args.audio_input
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = load_model(model_path)

    vocaliaztion_prediction(audio_path, model, device)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Classify chicken vocalization audio as Healthy/Noise/Unhealthy")
    parser.add_argument('--audio_input', type=str, required=True, help='Path to input audio file (.mp3 or .wav)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pth file)')
    args = parser.parse_args()

    main(args)