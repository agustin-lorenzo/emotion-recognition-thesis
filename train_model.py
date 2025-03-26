import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
import gc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize constants
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 10 # number of epochs for training

# data augmentation parameters, if needed
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 4 # how many augmented samples are created out of one original sample

# dataset object
class EEGDataset(Dataset):
    def __init__(self, trials, labels):
        self.trials = trials
        self.labels = labels
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return self.trials[idx], self.labels[idx]

# load data, create dataset/dataloader objects
print("Loading data...")
data = np.load('data/extracted_features.npz')
trials = data['trials']
labels = data['labels']
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
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=len(train_loader)*NUM_EPOCHS,
        pct_start=0.3
    )
    
    # record per-epoch train/validation losses
    train_epoch_losses = []
    val_epoch_losses = []
    
    # training loop
    vit.train()
    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
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
        print("Finished Training.")
        
        # record validation loss to ensure model is improving on unseen data
        print("\nValidating...")
        vit.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = vit(inputs)
                loss_val = loss_fn(outputs, targets)
                epoch_val_loss += loss_val.item() * inputs.size(0)
        epoch_val_loss /= len(test_set)
        val_epoch_losses.append(epoch_val_loss)
        
        print(f"Fold {fold_idx} Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss:.4f}", end='\r')
    print(f"\nFinished training fold {fold_idx}")
    
    # testing/validation loop
    print("\nTesting...")
    vit.eval()
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
        
        # classification metrics
        print("Computing metrics...")
        cm = confusion_matrix(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        try:
            roc_auc = roc_auc_score(label_binarize(all_targets, classes=range(4)), 
                                all_probs, average="macro", multi_class="ovr")
        except Exception as e:
            roc_auc = None
            print("ROC AUC computation error:", e)
        
        # save metrics
        fold_metrics.append({
            "fold": fold_idx,
            "final_train_loss": train_epoch_losses[-1],
            "final_val_loss": val_epoch_losses[-1],
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
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
        
        print(f"Fold {fold_idx} metrics:")
        print(f"\n  Train Loss (final epoch): {train_epoch_losses[-1]:.4f}")
        print(f"  Val Loss   (final epoch): {val_epoch_losses[-1]:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if roc_auc is not None:
            print(f"  ROC AUC:   {roc_auc:.4f}")
        else:
            print("  ROC AUC:   Could not compute")
        
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