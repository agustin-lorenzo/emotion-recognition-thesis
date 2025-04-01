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
from torch.utils.data import WeightedRandomSampler, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim import AdamW
import os
import warnings
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool3d(2)
#         self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.adaptive_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(32, num_classes)
        
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

# training hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 0 # set to 0 if one dataset object used for training/testing, otherwise 4
PIN_MEMORY = True
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01 # original weight decay: 0.01
GRADIENT_CLIP = 1.0 # max gradient norm allowed

# vit model parameters
VIT_IMAGE_PATCH = 16
VIT_FRAME_PATCH = 32
VIT_FRAMES = 256 # indicates current clip length being used
VIT_DIM = 768
VIT_DEPTH = 6
VIT_HEADS = 8
VIT_MLP_DIM = 1024
VIT_DROPOUT = 0.2
VIT_EMB_DROPOUT = 0.1
VIT_POOL = 'cls'

# data augmentation parameters, if needed
SNR = 5 # signal-to-noise ratio
REP_FACTOR = 2 # how many augmented samples are created out of one original sample
APPLY_NOISE_PROB = 0.5 # probability of applying noise to a sample



def compute_mean_std(trials, batch_size=1000):
    """
    Compute global mean and standard deviation in a memory-efficient way.
    
    trials: a memmapped (or large) numpy array.
    batch_size: Number of trials to process at a time.
    """
    n = len(trials)
    total_sum = 0.0
    total_count = 0
    # First pass: compute mean
    for i in range(0, n, batch_size):
        batch = trials[i:i+batch_size].astype(np.float32)
        total_sum += batch.sum()
        total_count += batch.size
    mean = total_sum / total_count

    # Second pass: compute variance
    total_sq_diff = 0.0
    for i in range(0, n, batch_size):
        batch = trials[i:i+batch_size].astype(np.float32)
        total_sq_diff += np.sum((batch - mean) ** 2)
    variance = total_sq_diff / total_count
    std = np.sqrt(variance)
    return mean, std

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
    #weights = 1. / np.sqrt(class_counts + 1e-8) # inverse square root, smoother
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    weights = weights / weights.sum()
    sample_weights = weights[labels_tensor]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels)*REP_FACTOR,
        replacement=True
    )
    return sampler

def get_metrics(y_true, y_pred, y_probs=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_probs is not None:
        try:
            if np.dim(y_probs) == 1:
                roc_auc = roc_auc_score(y_true, y_probs)
            else:
                
                roc_auc = roc_auc_score(label_binarize(y_true, classes=range(4)), 
                                     y_probs, average="macro", multi_class="ovr")
        except Exception as e:
            roc_auc = None
            print("ROC AUC computation error:", e)
    else:
        roc_auc = None
    return accuracy, precision, recall, f1, roc_auc

# dataset object
class EEGDataset(Dataset):
    def __init__(self, trials, labels, training=False, apply_noise_prob=APPLY_NOISE_PROB):
        self.trials = trials
        self.labels = labels
        self.training = training
        self.apply_noise_prob = apply_noise_prob
        # self.mean = np.mean(trials)
        # self.std = np.std(trials)
        self.mean, self.std = compute_mean_std(trials, batch_size=1000)
        
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]                # shape: (frames, height, width), dtype: uint8
        trial = np.expand_dims(trial, axis=0)     # shape: (1, frames, height, width)
        trial = (trial - self.mean) / (self.std + 1e-8)  # Normalize to [0, 1]
        trial = torch.from_numpy(trial).float()
        assert trial.shape == (1, 256, 128, 128), f"Unexpected input shape: {trial.shape}" # sanity checks
        #assert trial.min() >= 0 and trial.max() <= 1.0, "Input data not normalized properly"
        # add gaussian noise if training
        if self.training and random.random() < self.apply_noise_prob:
            trial = add_gaussian_noise_torch(trial, snr_db=SNR)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return trial, label

class DatasetRepeater(Dataset):
    def __init__(self, dataset, rep_factor=REP_FACTOR):
        self.dataset = dataset
        self.rep_factor = rep_factor

    def __len__(self):
        return len(self.dataset) * self.rep_factor

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        return self.dataset[idx]

# load data, create dataset/dataloader objects
print("Loading data...")
video_data = np.load('data/extracted_features_video.npz', mmap_mode='r')
picture_data = np.load('data/extracted_features_picture.npz', mmap_mode='r')
print("\tFile opened.")
picture_trials = picture_data['trials']
video_trials = video_data['trials']
print("\tTrials loaded.")
picture_labels = picture_data['labels'].astype(int)
video_labels = video_data['labels']
# Create dataset objects that wrap the memmapped arrays
video_dataset = EEGDataset(video_trials, video_labels, training=True)
picture_dataset = EEGDataset(picture_trials, picture_labels.astype(int), training=True)

# Use PyTorch's ConcatDataset to combine them without concatenating the underlying arrays
combined_dataset = ConcatDataset([video_dataset, picture_dataset])
combined_labels = np.concatenate((video_labels, picture_labels), axis=0)
print("\tCombined Shape:", len(combined_dataset))
# # trials = np.concatenate((video_trials, picture_trials), axis=0)
# labels = np.concatenate((video_labels, picture_labels), axis=0)
# print("\t\tShape:\n\t\t", video_trials.shape)
# print("\t\tShape:\n\t\t", picture_trials.shape)
# print("\tLabels loaded.")
# print("Data loaded.")

# load data, create dataset/dataloader objects # TODO: switch to above and use both datasets when good params are found
# print("Loading data...")
# data = np.load('data/extracted_features_video.npz', mmap_mode='r')
# print("\tFile opened.")
# trials = data['trials']
# print("\tTrials loaded.")
# print("\t\tShape:\n\t\t", trials.shape)
# labels = data['labels']
# print("\tLabels loaded.")
# print("Data loaded.")

# set up stratified 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []
epoch_loss_records = []
fold_idx = 1
for train_idx, test_idx in kf.split(np.zeros(len(combined_labels)), combined_labels):
    wandb.init(
    project="emotion-recognition",
    name=f"{VIT_FRAMES/128}sec_{LEARNING_RATE:.1e}lr",
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "vit_dim": VIT_DIM,
        "vit_depth": VIT_DEPTH,
        "vit_heads": VIT_HEADS,
        "vit_mlp_dim": VIT_MLP_DIM,
        "vit_dropout": VIT_DROPOUT,
        "vit_emb_dropout": VIT_EMB_DROPOUT,
        "vit_pool": VIT_POOL,
        "vit_image_patch": VIT_IMAGE_PATCH,
        "vit_frame_patch": VIT_FRAME_PATCH,
        "snr": SNR,
        "rep_factor": REP_FACTOR,
        "apply_noise_prob": APPLY_NOISE_PROB
    }
)
    print("============")
    print(f"|| Fold {fold_idx} ||")
    print("============")
    print("Fold class distribution:")
    print("  Training: ", np.bincount(combined_labels[train_idx])) 
    print("  Testing: ", np.bincount(combined_labels[test_idx]))
    print()
    
    # Use Subset to create train/test datasets for this fold
    train_subset = Subset(combined_dataset, train_idx)
    test_subset = Subset(combined_dataset, test_idx)
    
    # Optionally wrap the training subset with your DatasetRepeater
    train_dataset = DatasetRepeater(train_subset, rep_factor=REP_FACTOR)
    
    # Create a weighted sampler using the training fold labels
    train_sampler = create_weighted_sampler(combined_labels[train_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"Train Loader length: {len(train_loader)}")
    print(f"Test Loader length: {len(test_loader)}")
    
    # initialize model
    vit = ViT( # vision transformer (original) parameters as suggested by Awan et al. 2024
        image_size=128,
        frames=256, # = 1280 / 5 for 2 second clips
        image_patch_size=VIT_IMAGE_PATCH,
        frame_patch_size=VIT_FRAME_PATCH,
        num_classes=4,
        dim=VIT_DIM, # original: 768
        depth=VIT_DEPTH, # original: 16
        heads=VIT_HEADS, # original: 16
        mlp_dim=VIT_MLP_DIM, # original: 1024
        channels=1,
        dropout=VIT_DROPOUT, # original: 0.5
        emb_dropout=VIT_EMB_DROPOUT, # original: 0.1
        pool=VIT_POOL
    ).to(device)
    
    #vit = SimpleCNN(num_classes=4).to(device) # simple 3D CNN model
    
    # training hyperparameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(vit.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    scaler = torch.cuda.amp.GradScaler()
    max_lr = 1e-4
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=len(train_loader)*NUM_EPOCHS,
        pct_start=0.3,  # percent of total steps for LR warmup
        anneal_strategy='cos',  # cosine annealing for decaying
        div_factor=max_lr / LEARNING_RATE,  # determines initial LR: initial = max_lr/div_factor
        final_div_factor=1e4  # final LR = max_lr / (div_factor * final_div_factor)
    )
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
    warmup_epochs = 5 # number of warmup epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # start at 1% of the base LR
        total_iters=warmup_epochs 
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(NUM_EPOCHS * 0.7),  # Single cycle for majority of training
        eta_min=1e-6
    )
    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=4,
    #     verbose=True,
    #     min_lr=1e-6
    # )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\tEpoch: {epoch+1}\tLR: {current_lr:.2e}\tBatch Loss:{loss:.4f}\tCurrent grad. norm: {total_norm:.4f}\t", end='\r')
            torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step() # original scheduler location for one-cycle lr
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_dataset)
        train_epoch_losses.append(epoch_train_loss)
        #scheduler.step() # update lr per epoch for cosine annealing warm restarts
        
        # validation loop for each epoch
        print("\n\tValidation:")
        vit.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = vit(inputs)
                    loss_val = loss_fn(outputs, targets)
                epoch_val_loss += loss_val.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                correct_val += torch.sum(preds == targets).sum().item()
                total_val += targets.size(0)
        epoch_val_loss /= len(test_subset)
        # val_accuracy = correct_val / total_val
        val_acc, val_prec, val_rec, val_f1, val_roc_auc = get_metrics(all_targets, all_preds)
        val_epoch_losses.append(epoch_val_loss)
        roc_auc_str = f"{val_roc_auc:.4f}" if val_roc_auc is not None else "N/A"
        print(f"\tLoss: {epoch_val_loss:.4f} | Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | ROC AUC: {roc_auc_str}")
        print("\t:.......................................................................................................:")
        #scheduler.step(epoch_val_loss) # update lr per epoch for reduce on plateau based on val loss 
        scheduler.step() # or cosine annealing / sequential
        # if epoch < warmup_epochs:
        #     warmup_scheduler.step()
        # else:
        #     plateau_scheduler.step(epoch_val_loss)
        wandb.log({
            "fold": fold_idx,
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc,
            "lr": optimizer.param_groups[0]["lr"]
        })
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
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # calculate valence/arousal-specific metrics
        print("Computing metrics...")
        true_valence = (all_targets >= 2).astype(int)  # 0=low (0,1), 1=high (2,3)
        pred_valence = (all_preds >= 2).astype(int)
        valence_prob = all_probs[:, 2] + all_probs[:, 3]

        # Arousal: even vs odd (0/2 vs 1/3)
        true_arousal = (all_targets % 2).astype(int)  # 0=even (0,2), 1=odd (1,3)
        pred_arousal = (all_preds % 2).astype(int)
        arousal_prob = all_probs[:, 1] + all_probs[:, 3]
        
        valence_accuracy, valence_precision, valence_recall, valence_f1, valence_roc_auc = get_metrics(true_valence, pred_valence, valence_prob)
        
        arousal_accuracy, arousal_precision, arousal_recall, arousal_f1, arousal_roc_auc = get_metrics(true_arousal, pred_arousal, arousal_prob)
        # overall classification metrics
        overall_accuracy, overall_precision, overall_recall, overall_f1, overall_roc_auc = get_metrics(all_targets, all_preds, all_probs)
        cm = confusion_matrix(all_targets, all_preds)
        
        # save metrics
        fold_metrics.append({
            "fold": fold_idx,
            "final_train_loss": train_epoch_losses[-1],
            "final_val_loss": val_epoch_losses[-1],
            "confusion_matrix": cm,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_roc_auc": overall_roc_auc,
            "valence_accuracy": valence_accuracy,
            "valence_precision": valence_precision,
            "valence_recall": valence_recall,
            "valence_f1": valence_f1,
            "valence_roc_auc": valence_roc_auc,
            "arousal_accuracy": arousal_accuracy,
            "arousal_precision": arousal_precision,
            "arousal_recall": arousal_recall,
            "arousal_f1": arousal_f1,
            "arousal_roc_auc": arousal_roc_auc
        })
        
        wandb.log({
            "fold": fold_idx,
            "final_train_loss": train_epoch_losses[-1],
            "final_val_loss": val_epoch_losses[-1],
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_roc_auc": overall_roc_auc,
            "valence_accuracy": valence_accuracy,
            "valence_precision": valence_precision,
            "valence_recall": valence_recall,
            "valence_f1": valence_f1,
            "valence_roc_auc": valence_roc_auc,
            "arousal_accuracy": arousal_accuracy,
            "arousal_precision": arousal_precision,
            "arousal_recall": arousal_recall,
            "arousal_f1": arousal_f1,
            "arousal_roc_auc": arousal_roc_auc
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
        print(f"  Accuracy:          {overall_accuracy:.4f}")
        print(f"  Precision:         {overall_precision:.4f}")
        print(f"  Recall:            {overall_recall:.4f}")
        print(f"  F1 Score:          {overall_f1:.4f}")
        overall_roc_auc = f"{overall_roc_auc:.4f}" if overall_roc_auc is not None else "N/A"
        print(f"  ROC AUC:           {overall_roc_auc}")

        # print valence metrics
        print("\nValence Metrics:")
        print(f"  Accuracy:          {valence_accuracy:.4f}")
        print(f"  Precision:         {valence_precision:.4f}")
        print(f"  Recall:            {valence_recall:.4f}")
        print(f"  F1 Score:          {valence_f1:.4f}")
        valence_roc_auc = f"{valence_roc_auc:.4f}" if valence_roc_auc is not None else "N/A"
        print(f"  ROC AUC:           {valence_roc_auc}")

        # print arousal metrics
        print("\nArousal Metrics:")
        print(f"  Accuracy:          {arousal_accuracy:.4f}")
        print(f"  Precision:         {arousal_precision:.4f}")
        print(f"  Recall:            {arousal_recall:.4f}")
        print(f"  F1 Score:          {arousal_f1:.4f}")
        arousal_roc_auc = f"{arousal_roc_auc:.4f}" if arousal_roc_auc is not None else "N/A"
        print(f"  ROC AUC:           {arousal_roc_auc}")
        
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