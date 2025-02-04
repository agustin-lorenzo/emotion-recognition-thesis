import fcwt
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# helper method for adding gaussian noise to each channel
def add_gaussian_noise(signal, snr_db=5):
    """
    Add Gaussian noise to signal with specified SNR (dB)
    Returns noisy signal in same 0-255 uint8 format
    """
    # Convert to float for calculations
    signal_float = signal.astype(np.float32)
    
    # Calculate signal power (mean squared)
    signal_power = np.mean(signal_float ** 2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise with calculated power
    noise = np.random.normal(0, math.sqrt(noise_power), signal.shape)
    
    # Add noise and maintain original range
    noisy_signal = signal_float + noise
    noisy_signal = np.clip(noisy_signal, 0, 255)
    
    return noisy_signal.astype(np.uint8)

# load dataset
x = pickle.load(open('data_preprocessed_python/s01.dat', 'rb'), encoding='latin1')
data = x['data']
labels = x['labels']

relevant_channels = data[:, :32, :]
relevant_labels = labels[:, :2]

classes = []
for trial in range(40):
    # 4 valence-arousal classes
    valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
    cls = 0 if valence < 4.5 and arousal < 4.5 else \
                    1 if valence < 4.5 else \
                    2 if arousal < 4.5 else 3
    classes.append(cls)

# initialize constant variables
# paramaters for calculating cwt, not to be changed
fs = 128 
f0 = 4 # lowest frequency
f1 = 45 # highest frequency
fn = 32 # number of frequencies, match channel number for square frame

# samples and the target class for each sample, these two lists make up the dataset used by the model
six_sec_samples = []
sample_classes = []

print("Creating frames from EEG data...")
for trial in range(40):
    total_cwt = np.zeros((1024, 8064))

    for channel in range(32):    
        signal = relevant_channels[trial][channel]
        _, current_cwt = fcwt.cwt(signal, fs, f0, f1, fn)
        start = channel * fn
        end = (channel + 1) * fn
        total_cwt[start:end, :] = abs(current_cwt)
    
    # convert 2D time sample x channel-frequency format (x, y) to 3D channel x frequency x time format (x, y, z)
    # each 'frame' in all_frames is a 2D image showing each channel's CWT value for that time sample
    all_frames = []

    for sample in range(8064):
        frame = np.zeros((32, 32))
        for channel in range(32):
            start = channel * fn
            end = (channel + 1) * fn
            frame[:, channel] = total_cwt[start:end, sample]

        # normalize frame
        norm_frame = (frame - frame.min()) / (frame.max() - frame.min())
        # scale to 0-255
        scaled_frame = (norm_frame * 255).astype(np.uint8)

        # stack for RGB
        frame_rgb = np.stack((scaled_frame,) * 3, axis=0)
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        all_frames.append(frame_rgb)
        #all_frames.append(frame)
    
    # split trial into 6-second samples as suggested in Arjun et al. 2021
    window_size = 6 * 128
    for i in range(0, 8064, window_size):
        if i + window_size > 8064:
            break
        original_sample = all_frames[i:i+window_size]
        six_sec_samples.append(original_sample)
        sample_classes.append(classes[trial])
        
        # create 20 augmented copies from original frame as suggested in Li et al. 2016
        for _ in range(9):
            augmented_sample = []
            for frame in original_sample:
                noisy_frame = np.stack([
                    add_gaussian_noise(frame[c], snr_db=5)
                    for c in range(3)  # processing each "RGB" channel
                ], axis=0)
                augmented_sample.append(noisy_frame)
            
            six_sec_samples.append(augmented_sample)
            sample_classes.append(classes[trial])
print("All frames and copies initialized.")

class CWTDataset(Dataset):
    def __init__(self, data, labels):
        self.original_length = 768  # Hardcoded temporal length
        self.data = torch.tensor(np.array(data), dtype=torch.float32).permute(0, 2, 1, 3, 4)
        
        # Normalization (entire dataset)
        self.data = self.data / 255.0
        self.mean = self.data.mean(dim=(0, 2, 3, 4), keepdim=True)
        self.std = self.data.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
        self.data = (self.data - self.mean) / self.std
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.training = False  # Default mode

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.data[index].clone()
        label = self.labels[index]
        
        if self.training:
            # Gaussian noise
            if random.random() > 0.5:
                x += torch.randn_like(x) * 0.05 * random.random()
        
        return x, label
    

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self
    

print("\nCreating dataset objects...")    
train_data, test_data, train_labels, test_labels = train_test_split(
    six_sec_samples, sample_classes, test_size=0.2, random_state=23, stratify=sample_classes
)
#train_dataset = CWTDataset(train_data, train_labels).train()
train_loader = DataLoader(CWTDataset(train_data, train_labels).train(), batch_size=4, shuffle=True)
test_loader = DataLoader(CWTDataset(test_data, test_labels).eval(), batch_size=4)
print("Dataset objects initialized.")

print(f"Train dataset shape:\n\t{np.shape(train_data)}")
print(f"Test dataset shape:\n\t{np.shape(test_data)}")

vit = ViT(
    image_size=32,
    frames=768,
    image_patch_size=16,
    frame_patch_size=96,
    num_classes=4,
    dim=768,
    depth=16,
    heads=16,
    mlp_dim=1024,
    channels=3,
    dropout=0.5,
    emb_dropout=0.1,
    pool='mean'
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3e-4,
    total_steps=len(train_loader)*50,
    pct_start=0.3
)

print("\nTraining...")
for epoch in range(50):
    vit.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = loss_fn(outputs, targets)
        print(f"\tEpoch: {epoch}\tLoss:{loss:.4f}", end='\r')
        loss.backward()
        optimizer.step()
    scheduler.step()
print("\nFinished Training.")

vit.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = vit(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%")

torch.save(vit.state_dict(), "models/s01model.pth")