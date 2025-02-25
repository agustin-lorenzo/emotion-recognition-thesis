import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
import random
import csv
import copy
import gc
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize constants
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 1 # how many augmented samples are created out of one original sample
NUM_EPOCHS = 50 # number of epochs for training

csv_file = f'cross_subject_model_metrics_{NUM_EPOCHS}epochs.csv'
header = ["subject", "valence_accuracy", "arousal_accuracy", "overall_accuracy"]

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# helper method for adding gaussian noise to each frame in a sample, includes signal-to-noise parameter
# altered to handle pytorch tensors rather than numpy array
def add_gaussian_noise_torch(signal_tensor, snr_db=5):
    signal_power = (signal_tensor ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(signal_tensor) * noise_std
    return signal_tensor + noise

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
                    #x += torch.randn_like(x) * 0.05 * random.random()
                    augmented_sample = []
                    for frame in x:
                        noisy_frame = copy.deepcopy(frame)
                        for channel in range(32):
                            noisy_frame[:, :, channel] = add_gaussian_noise_torch(noisy_frame[:, :, channel], SNR)
                        augmented_sample.append(noisy_frame)

                    augmented_tensor = torch.stack(augmented_sample, dim=0)
                    return augmented_tensor, label
                
            return x, label

        def train(self):
            self.training = True
            return self

        def eval(self):
            self.training = False
            return self

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


# model training
print("Loading training data...")
train_file = np.load("test_data.npz", allow_pickle=True) # read train data from disk
train_data = train_file["samples"]
train_labels = train_file["labels"]
train_loader = DataLoader(CWTDataset(train_data, train_labels).train(), batch_size=4, shuffle=True)
print("Training data initialized.")
print(f"\nTrain dataset shape:\n\t{np.shape(train_data)}")

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
del train_loader, train_data, train_labels, train_file # free up space
gc.collect()


# model testing
vit.eval()
valence_correct = 0 # valence-specific accuracy
arousal_correct = 0 # arousal-specific accuracy
overall_correct = 0 # overall accuracy in classifying accross 4 possible cases
total = 0

print("Loading testing data...")
test_file = np.load("test_data.npz", allow_pickle=True) # read train data from disk
test_data = test_file["samples"]
test_labels = test_file["labels"]
test_loader = DataLoader(CWTDataset(test_data, test_labels).eval(), batch_size=4)
print("Testing data initalized.")
print(f"Test dataset shape:\n\t{np.shape(test_data)}")

# lists for confusion matrix
pred = []
y_test = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = vit(inputs)
        _, predicted = torch.max(outputs, 1)
        y_test.append(targets)
        pred.append(predicted)

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
    writer.writerow(["all", valence_accuracy, arousal_accuracy, overall_accuracy])
    file.flush()  # flush ensures the row is written immediately

torch.save(vit.state_dict(), f"models/cross_subject_model_{NUM_EPOCHS}epochs.pth")

# creating confusion matrix
cm = confusion_matrix(y_test, pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv("cm_df.csv", index=False)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()