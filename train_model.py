import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
import gc
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# initialize constants
BATCH_SIZE = 2
NUM_WORKERS = 0 # set to 0 if on-the-fly augmentation, otherwise 4
PIN_MEMORY = True
NUM_EPOCHS = 50 # number of epochs for training
LEARNING_RATE = 3e-4

# data augmentation parameters, if needed
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 4 # how many augmented samples are created out of one original sample
APPLY_NOISE_PROB = 0.2 # probability of applying noise to a sample

# helper method for adding gaussian noise to each frame in a sample, includes signal-to-noise parameter
# altered to handle pytorch tensors rather than numpy array
def add_gaussian_noise_torch(signal_tensor, snr_db=5):
    signal_power = (signal_tensor ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(signal_tensor) * noise_std
    return signal_tensor + noise

# dataset object
class EEGDataset(Dataset):
    def __init__(self, trials, labels, training=False, apply_noise_prob=APPLY_NOISE_PROB):
        self.trials = trials
        self.labels = labels
        self.training = training
        self.apply_noise_prob = apply_noise_prob
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]                # shape: (frames, height, width), dtype: uint8
        trial = np.expand_dims(trial, axis=0)     # shape: (1, frames, height, width)
        trial = torch.from_numpy(trial).float()   # convert to float32 tensor
        # add gaussian noise if training
        if self.training and random.random() < self.apply_noise_prob:
            for frame in range(trial.size(1)):
                trial[:, frame] = add_gaussian_noise_torch(trial[:, frame], snr_db=SNR)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return trial, label

# load data, create dataset/dataloader objects
print("Loading data...")
data = np.load('data/extracted_features_compressed.npz')
print("\tFile opened.")
trials = data['trials']
print("\tTrials loaded.")
print("\tShape:\n\t", trials.shape)
labels = data['labels']
print("\tLabels loaded.")
dataset = EEGDataset(trials, labels)
print("Data loaded.")

# set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []
epoch_loss_records = []
fold_idx = 1
for train_idx, test_idx in kf.split(dataset):
    print("----------------------------------------")
    print(f"Fold {fold_idx}")
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # initialize model
    vit = ViT( # vision transformer (original) parameters as suggested by Awan et al. 2024
        image_size=128,
        frames=1280,
        image_patch_size=32,
        frame_patch_size=32,
        num_classes=4,
        dim=768, # original: 768
        depth=16, # original: 16
        heads=16, # original: 16
        mlp_dim=1024, # original: 1024
        channels=1,
        dropout=0.5,
        emb_dropout=0.1,
        pool='mean'
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01) # original weight decay: 0.01
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-5,
        total_steps=len(train_loader)*NUM_EPOCHS,
        pct_start=0.3
    )
    
    # record per-epoch train/validation losses
    train_epoch_losses = []
    val_epoch_losses = []
    
    # training loop
    print()
    for epoch in range(NUM_EPOCHS):
        print(f"Training epoch {epoch + 1}:")
        vit.train()
        dataset.training = True
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = vit(inputs)
                loss = loss_fn(outputs, targets)
            print(f"\tEpoch: {epoch+1}\tCurrent Loss:{loss}", end='\r')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_set)
        train_epoch_losses.append(epoch_train_loss)
        
        # record validation loss to ensure model is improving on unseen data
        print("\n\tValidation:")
        vit.eval()
        dataset.training = False
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = vit(inputs)
                loss_val = loss_fn(outputs, targets)
                epoch_val_loss += loss_val.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct_val += torch.sum(preds == targets).sum().item()
                total_val += targets.size(0)
        epoch_val_loss /= len(test_set)
        val_accuracy = correct_val / total_val
        val_epoch_losses.append(epoch_val_loss)
        print(f"\tLoss: {epoch_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")
    print(f"\nFinished training fold {fold_idx}")
    
    # testing/validation loop
    print("\nTesting...")
    vit.eval()
    dataset.training = False
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vit(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # calculate valence/arousal-specific metrics
        print("Computing metrics...")
        true_valence = np.array([0 if t in (0, 1) else 1 for t in all_targets])
        pred_valence = np.array([0 if p in (0, 1) else 1 for p in all_preds])

        true_arousal = np.array([0 if t in (0, 2) else 1 for t in all_targets])
        pred_arousal = np.array([0 if p in (0, 2) else 1 for p in all_preds])
        
        valence_accuracy = np.mean(true_valence == pred_valence)
        valence_precision = precision_score(true_valence, pred_valence, average='binary')
        valence_recall = recall_score(true_valence, pred_valence, average='binary')
        valence_f1 = f1_score(true_valence, pred_valence, average='binary')
        valence_prob = all_probs[:, 2] + all_probs[:, 3]
        try:
            roc_auc_valence = roc_auc_score(true_valence, valence_prob)
        except Exception as e:
            roc_auc_valence = None
            print("Valence ROC AUC computation error:", e)

        arousal_accuracy = np.mean(true_arousal == pred_arousal)
        arousal_precision = precision_score(true_arousal, pred_arousal, average='binary')
        arousal_recall = recall_score(true_arousal, pred_arousal, average='binary')
        arousal_f1 = f1_score(true_arousal, pred_arousal, average='binary')
        arousal_prob = all_probs[:, 1] + all_probs[:, 3]
        try:
            roc_auc_arousal = roc_auc_score(true_arousal, arousal_prob)
        except Exception as e:
            roc_auc_arousal = None
            print("Arousal ROC AUC computation error:", e)
        
        # overall classification metrics
        accuracy = np.mean(all_preds == all_targets)
        cm = confusion_matrix(all_targets, all_preds)
        precision_overall = precision_score(all_targets, all_preds, average='macro')
        recall_overall = recall_score(all_targets, all_preds, average='macro')
        f1_overall = f1_score(all_targets, all_preds, average='macro')
        try:
            roc_auc_overall = roc_auc_score(label_binarize(all_targets, classes=range(4)), 
                                all_probs, average="macro", multi_class="ovr")
        except Exception as e:
            roc_auc_overall = None
            print("ROC AUC computation error:", e)
        
        # save metrics
        fold_metrics.append({
            "fold": fold_idx,
            "final_train_loss": train_epoch_losses[-1],
            "final_val_loss": val_epoch_losses[-1],
            "confusion_matrix": cm,
            "overall_accuracy": accuracy,
            "overall_precision": precision_overall,
            "overall_recall": recall_overall,
            "overall_f1": f1_overall,
            "overall_roc_auc": roc_auc_overall,
            "valence_accuracy": valence_accuracy,
            "valence_precision": valence_precision,
            "valence_recall": valence_recall,
            "valence_f1": valence_f1,
            "valence_roc_auc": roc_auc_valence,
            "arousal_accuracy": arousal_accuracy,
            "arousal_precision": arousal_precision,
            "arousal_recall": arousal_recall,
            "arousal_f1": arousal_f1,
            "arousal_roc_auc": roc_auc_arousal
        })
        
        # save loss data
        for epoch in range(NUM_EPOCHS):
            epoch_loss_records.append({
                "fold": fold_idx,
                "epoch": epoch+1,
                "train_loss": train_epoch_losses[epoch],
                "val_loss": val_epoch_losses[epoch]
            })
        print("Metrics recorded.")
        
        # Print overall metrics
        print("Overall Metrics:")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  Precision:         {precision_overall:.4f}")
        print(f"  Recall:            {recall_overall:.4f}")
        print(f"  F1 Score:          {f1_overall:.4f}")
        if roc_auc_overall is not None:
            print(f"  ROC AUC:           {roc_auc_overall:.4f}")
        else:
            print("  ROC AUC:           Could not compute")

        # Print valence metrics
        print("\nValence Metrics:")
        print(f"  Accuracy:          {valence_accuracy:.4f}")
        print(f"  Precision:         {valence_precision:.4f}")
        print(f"  Recall:            {valence_recall:.4f}")
        print(f"  F1 Score:          {valence_f1:.4f}")
        if roc_auc_valence is not None:
            print(f"  ROC AUC:           {roc_auc_valence:.4f}")
        else:
            print("  ROC AUC:           Could not compute")

        # Print arousal metrics
        print("\nArousal Metrics:")
        print(f"  Accuracy:          {arousal_accuracy:.4f}")
        print(f"  Precision:         {arousal_precision:.4f}")
        print(f"  Recall:            {arousal_recall:.4f}")
        print(f"  F1 Score:          {arousal_f1:.4f}")
        if roc_auc_arousal is not None:
            print(f"  ROC AUC:           {roc_auc_arousal:.4f}")
        else:
            print("  ROC AUC:           Could not compute")
        
        fold_idx += 1
        del vit, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

# save metrics to csv
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_df.to_csv("fold_metrics.csv", index=False)
print(f"\nFold summary metrics saved.")

# save loss data to csv
epoch_loss_df = pd.DataFrame(epoch_loss_records)
epoch_loss_csv_path = "epoch_losses.csv"
epoch_loss_df.to_csv(epoch_loss_csv_path, index=False)
print(f"Epoch loss records saved.")