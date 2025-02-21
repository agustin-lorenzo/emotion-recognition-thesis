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
import csv
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize constants
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 1 # how many augmented samples are created out of one original sample
NUM_EPOCHS = 20 # number of epochs for training

csv_file = f'cross_subject_model_metrics_{NUM_EPOCHS}epochs.csv'
header = ["subject", "valence_accuracy", "arousal_accuracy", "overall_accuracy"]

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# dataset object for CWT data
class CWTDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(np.array(data), dtype=torch.float32).permute(0, 2, 1, 3, 4)
            # normalize data over entire dataset
            self.data = self.data / 255.0
            self.mean = self.data.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.std = self.data.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
            self.data = (self.data - self.mean) / self.std
            
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.training = False

        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            x = self.data[index].clone()
            label = self.labels[index]
            
            if self.training: # extra Gaussian noise during training
                if random.random() > 0.5:
                    x += torch.randn_like(x) * 0.05 * random.random()
            return x, label

        def train(self):
            self.training = True
            return self

        def eval(self):
            self.training = False
            return self

six_sec_samples = [] # all 6-second samples from all subjects in the dataset
sample_classes = []  # all target classes for each of the 6-second samples

for subject in range(32):
    file_path = f"data_preprocessed_python/s{subject + 1:02}.dat"
    x = pickle.load(open(file_path, 'rb'), encoding='latin1')
    data = x['data']
    labels = x['labels']
    relevant_channels = data[:, :32, :]
    relevant_labels = labels[:, :2]
    classes = []

    for trial in range(40):
        # 4 valence-arousal classes
        valence, arousal = relevant_labels[trial][0], relevant_labels[trial][1]
        if valence < 4.5 and arousal < 4.5: # LVLA
            cls = 0
        elif valence < 4.5:                 # LVHA
            cls = 1
        elif arousal < 4.5:                 # HVLA
            cls = 2
        else:                               # HVHA
            cls = 3
        classes.append(cls)

    # initialize constant variables
    # paramaters for calculating cwt, not to be changed
    fs = 128 
    f0 = 4 # lowest frequency
    f1 = 45 # highest frequency
    fn = 32 # number of frequencies, match channel number for square frame

    print(f"Creating frames from participant {subject + 1}'s EEG data...")
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
            #all_frames.append(frame) # append 1-channel frame without rgb
        
        # split trial into 6-second samples as suggested in Arjun et al. 2021
        window_size = 6 * 128
        for i in range(0, 8064, window_size):
            if i + window_size > 8064:
                break
            original_sample = all_frames[i:i+window_size]
            six_sec_samples.append(original_sample)
            sample_classes.append(classes[trial])

print("All frames and copies initialized.")
print("\nCreating dataset objects...")    
train_data, test_data, train_labels, test_labels = train_test_split(
    six_sec_samples, sample_classes, test_size=0.2, random_state=23, stratify=sample_classes
)
train_loader = DataLoader(CWTDataset(train_data, train_labels).train(), batch_size=4, shuffle=True)
test_loader = DataLoader(CWTDataset(test_data, test_labels).eval(), batch_size=4)
print("Dataset objects initialized.")
print(f"\nTrain dataset shape:\n\t{np.shape(train_data)}")
print(f"Test dataset shape:\n\t{np.shape(test_data)}")

vit = ViT( # vision transformer parameters as suggested by Awan et al. 2024
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

loss_fn = nn.CrossEntropyLoss() # cross entropy is appropriate for classification
optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3e-4,
    total_steps=len(train_loader)*NUM_EPOCHS,
    pct_start=0.3
)

print("\nTraining...")
for epoch in range(NUM_EPOCHS):
    vit.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = loss_fn(outputs, targets)
        print(f"\tEpoch: {epoch+1}\tLoss:{loss}", end='\r')
        loss.backward()
        optimizer.step()
    scheduler.step()
print("\nFinished Training.")

vit.eval()
valence_correct = 0 # valence-specific accuracy
arousal_correct = 0 # arousal-specific accuracy
overall_correct = 0 # overall accuracy in classifying accross 4 possible cases
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = vit(inputs)
        _, predicted = torch.max(outputs, 1)

        # getting number of correct instances over all instances
        total += targets.size(0) 
        for i in range(len(targets)):
            valence_correct += (predicted[i] in (0, 1) and targets[i] in (0, 1)) # both low valence
            valence_correct += (predicted[i] in (2, 3) and targets[i] in (2, 3)) # both high valence
            arousal_correct += (predicted[i] in (0, 2) and targets[i] in (0, 2)) # both low arousal
            arousal_correct += (predicted[i] in (1, 3) and targets[i] in (1, 3)) # both high arousal
        overall_correct += (predicted == targets).sum().item()

# calculating and reporting accuracy metrics
valence_accuracy = 100 * valence_correct / total
arousal_accuracy = 100 * arousal_correct / total
overall_accuracy = 100 * overall_correct / total
print(f"\nValence accuracy: {valence_accuracy:.4f}%")
print(f"Arousal accuracy: {arousal_accuracy:.4f}%")
print(f"Overall accuracy: {overall_accuracy:.4f}%")
print("---------------------------------------")
# save metrics to .csv
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([subject + 1, valence_accuracy, arousal_accuracy, overall_accuracy])
    file.flush()  # flush ensures the row is written immediately

torch.save(vit.state_dict(), f"models/cross_subject_model.pth")