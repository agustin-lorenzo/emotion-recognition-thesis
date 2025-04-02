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
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

SNR = 5 # signal-to-noise ratio
REP_FACTOR = 4 # how many augmented samples are created out of one original sample
NUM_EPOCHS = 100 # number of epochs for training

csv_file = f'cross_subject_model_metrics_{REP_FACTOR}augmented_{NUM_EPOCHS}epochs.csv'
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

def apply_gpu_noise(inputs, snr_db=5, prob=0.8):
    if random.random() < prob:
        for b in range(inputs.size(0)):       # each sample in batch
            for frame_idx in range(inputs.size(1)):  # 768 frames
                for col in range(32):  # each column
                    inputs[b, frame_idx, :, :, col] = add_gaussian_noise_torch(
                        inputs[b, frame_idx, :, :, col],
                        snr_db
                    )
    return inputs

# dataset object for CWT data
class OnTheFlyDataset(Dataset):
    def __init__(self, data_file, training=False, snr_db=5, apply_noise_prob=0.8): # TODO: remove params if gpu-augment works
        self.npz_obj = np.load(data_file, mmap_mode='r')
        self.samples = self.npz_obj['samples'] 
        self.labels = self.npz_obj['labels']   
        self.training = training
        # self.snr_db = snr_db # commenting out to try gpu-gaussian noise
        # self.apply_noise_prob = apply_noise_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Lazy-load a single sample from disk (shape: (768, 3, 32, 32))
        sample_np = self.samples[idx]
        label = self.labels[idx]
        sample_tensor = torch.from_numpy(sample_np).float()
        
        sample_tensor /= 255.0
        mean = sample_tensor.mean()
        std = sample_tensor.std() + 1e-8
        sample_tensor = (sample_tensor - mean) / std

        # # If training, optionally apply noise # commenting out to try gpu-gaussian noise
        # if self.training:
        #     if random.random() < self.apply_noise_prob:
        #         for frame_idx in range(sample_tensor.size(0)):   # 768 frames
        #             frame = sample_tensor[frame_idx]
        #             for col in range(32):  # each column
        #                 frame[:, :, col] = add_gaussian_noise_torch(frame[:, :, col], self.snr_db)

        sample_tensor = sample_tensor.permute(1, 0, 2, 3) # reorder sample for correct vit input
        return sample_tensor, torch.tensor(label, dtype=torch.long)



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
train_dataset = OnTheFlyDataset(
    data_file="train_data.npz",   # must be uncompressed
    training=True,
    snr_db=SNR,
    apply_noise_prob=0.5
)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY)
print("Training data initialized.")
print(f"Training set size: {len(train_dataset)} samples")

loss_fn = nn.CrossEntropyLoss() # cross entropy is appropriate for classification
optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
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
        with torch.cuda.amp.autocast():
            outputs = vit(inputs)
            loss = loss_fn(outputs, targets)
        print(f"\tEpoch: {epoch+1}\tLoss:{loss}", end='\r')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
print("\nFinished Training.")
del train_loader, train_dataset
gc.collect()


# model testing
print("Loading testing data...")
test_dataset = OnTheFlyDataset(
    data_file="test_data.npz",    # must be uncompressed
    training=False
)
test_loader = DataLoader(test_dataset,
                        shuffle=True,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY)
print("Testing data initalized.")
print(f"Testing set size: {len(test_dataset)} samples")

vit.eval()
valence_correct = 0 # valence-specific accuracy
arousal_correct = 0 # arousal-specific accuracy
overall_correct = 0 # overall accuracy in classifying accross 4 possible cases
total = 0

# lists for confusion matrix
y_test = []
pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        inputs = apply_gpu_noise(inputs, snr_db=SNR, prob=0.8)
        outputs = vit(inputs)
        _, predicted = torch.max(outputs, 1)
       
        y_test.append(targets.cpu()) # move tensors to cpu and append to saved list
        pred.append(predicted.cpu())

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

# save model
torch.save(vit.state_dict(), f"models/cross_subject_model_{REP_FACTOR}augmented_{NUM_EPOCHS}epochs.pth")

# creating confusion matrix
class_names = ["LVLA", "LVHA", "HVLA", "HVHA"]

pred_cpu = torch.cat(pred) # converting test/pred lists to numpy array format for confusion matrix
y_test_cpu = torch.cat(y_test)
y_test_np = y_test_cpu.numpy()
pred_np = pred_cpu.numpy()

cm = confusion_matrix(y_test_np, pred_np)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(f"cm_df_{REP_FACTOR}augmented_{NUM_EPOCHS}epochs.csv", index=False)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()