import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
import gc
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# initialize constants
BATCH_SIZE = 16
NUM_WORKERS = 0 # set to 0 if on-the-fly augmentation, otherwise 4
PIN_MEMORY = True
NUM_EPOCHS = 50 # number of epochs for training
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01 # original weight decay: 0.01

# data augmentation parameters, if needed
SNR = 10 # signal-to-noise ratio
REP_FACTOR = 4 # how many augmented samples are created out of one original sample
APPLY_NOISE_PROB = 0.25 # probability of applying noise to a sample

# helper method for adding gaussian noise to each frame in a sample, includes signal-to-noise parameter
# altered to handle pytorch tensors rather than numpy array
def add_gaussian_noise_torch(signal_tensor, snr_db=5):
    signal_power = (signal_tensor ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(signal_tensor) * noise_std
    return signal_tensor + noise

# helper method for creating a weighted sampler to handle class imbalance
def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts + 1e-8, dtype=torch.float)
    weights = weights / weights.sum()
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler

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
        trial = torch.from_numpy(trial).float() / 255.0  # convert to float32 tensor, normalize to [0, 1]
        # add gaussian noise if training
        if self.training and random.random() < self.apply_noise_prob:
            trial = add_gaussian_noise_torch(trial, snr_db=SNR)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return trial, label

# load data, create dataset/dataloader objects
print("Loading data...")
data = np.load('data/extracted_features_compressed.npz')
print("\tFile opened.")
trials = data['trials']
print("\tTrials loaded.")
print("\t\tShape:\n\t\t", trials.shape)
labels = data['labels'].astype(int) #TODO: save as int in original feature extraction
print("\tLabels loaded.")
print("Data loaded.")

# set up stratified 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []
epoch_loss_records = []
fold_idx = 1
for train_idx, test_idx in kf.split(trials, labels):
    print("=====================================")
    print(f"|| Fold {fold_idx} ||")
    print("============")
    # get class weights for weighted loss function
    train_labels = labels[train_idx]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Fold class distribution:")
    print("  Training: ", np.bincount(labels[train_idx])) 
    print("  Testing: ", np.bincount(labels[test_idx]))
    print()
    
    # initalize datasets w/ sampler for fold
    train_sampler = create_weighted_sampler(labels[train_idx])
    train_dataset = EEGDataset(trials[train_idx], labels[train_idx], training=True)
    test_dataset = EEGDataset(trials[test_idx], labels[test_idx], training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # initialize model
    vit = ViT( # vision transformer (original) parameters as suggested by Awan et al. 2024
        image_size=128,
        frames=256, # = 1280 / 5 for 2 second clips
        image_patch_size=16,
        frame_patch_size=32,
        num_classes=4,
        dim=768, # original: 768
        depth=8, # original: 16
        heads=8, # original: 16
        mlp_dim=1024, # original: 1024
        channels=1,
        dropout=0.1, # original: 0.5
        emb_dropout=0.1, # original: 0.1
        pool='cls'
    ).to(device)
    
    # training hyperparameters
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(vit.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    scaler = torch.cuda.amp.GradScaler()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR( # original scheduler
    #     optimizer,
    #     max_lr=LEARNING_RATE * 2,
    #     total_steps=len(train_loader)*NUM_EPOCHS,
    #     pct_start=0.3
    # )
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=25,         # Number of epochs per restart cycle
    #     T_mult=1,        # Cycle length multiplier (double after each cycle)
    #     eta_min=1e-6,    # Minimum learning rate (1% of initial LR)
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # record per-epoch train/validation losses
    train_epoch_losses = []
    val_epoch_losses = []
    
    # training loop
    for epoch in range(NUM_EPOCHS):
        print()
        vit.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = vit(inputs)
                loss = loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = 0.0
            for p in vit.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"\tEpoch: {epoch+1}\tCurrent Loss:{loss}\t   Gradient norm: {total_norm:.4f}", end='\r')
            torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step() # original scheduler location for one-cycle lr
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_dataset)
        train_epoch_losses.append(epoch_train_loss)
        #scheduler.step() # update lr per epoch for cosine annealing
        
        # validation loop for each epoch
        print("\n\tValidation:")
        vit.eval()
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
        epoch_val_loss /= len(test_dataset)
        val_accuracy = correct_val / total_val
        val_epoch_losses.append(epoch_val_loss)
        print(f"\t\tVal Loss: {epoch_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")
        scheduler.step(epoch_val_loss)
    print(f"\nFinished training fold {fold_idx}")
    
    # testing loop
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
        
        # calculate valence/arousal-specific metrics
        print("Computing metrics...")
        true_valence = (all_targets >= 2).astype(int)  # 0=low (0,1), 1=high (2,3)
        pred_valence = (all_preds >= 2).astype(int)

        # Arousal: even vs odd (0/2 vs 1/3)
        true_arousal = (all_targets % 2).astype(int)  # 0=even (0,2), 1=odd (1,3)
        pred_arousal = (all_preds % 2).astype(int)
        
        valence_accuracy = np.mean(true_valence == pred_valence)
        valence_precision = precision_score(true_valence, pred_valence, average='binary', zero_division=0)
        valence_recall = recall_score(true_valence, pred_valence, average='binary', zero_division=0)
        valence_f1 = f1_score(true_valence, pred_valence, average='binary', zero_division=0)
        valence_prob = all_probs[:, 2] + all_probs[:, 3]
        try:
            roc_auc_valence = roc_auc_score(true_valence, valence_prob)
        except Exception as e:
            roc_auc_valence = None
            print("Valence ROC AUC computation error:", e)

        arousal_accuracy = np.mean(true_arousal == pred_arousal)
        arousal_precision = precision_score(true_arousal, pred_arousal, average='binary', zero_division=0)
        arousal_recall = recall_score(true_arousal, pred_arousal, average='binary', zero_division=0)
        arousal_f1 = f1_score(true_arousal, pred_arousal, average='binary', zero_division=0)
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
        
        # print overall metrics
        print("Overall Metrics:")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  Precision:         {precision_overall:.4f}")
        print(f"  Recall:            {recall_overall:.4f}")
        print(f"  F1 Score:          {f1_overall:.4f}")
        if roc_auc_overall is not None:
            print(f"  ROC AUC:           {roc_auc_overall:.4f}")
        else:
            print("  ROC AUC:           Could not compute")

        # print valence metrics
        print("\nValence Metrics:")
        print(f"  Accuracy:          {valence_accuracy:.4f}")
        print(f"  Precision:         {valence_precision:.4f}")
        print(f"  Recall:            {valence_recall:.4f}")
        print(f"  F1 Score:          {valence_f1:.4f}")
        if roc_auc_valence is not None:
            print(f"  ROC AUC:           {roc_auc_valence:.4f}")
        else:
            print("  ROC AUC:           Could not compute")

        # print arousal metrics
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