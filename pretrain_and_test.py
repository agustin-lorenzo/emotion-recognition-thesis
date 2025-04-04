import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vit_pytorch.vit_3d import ViT
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import csv
import gc
import warnings
import h5py
import wandb
from tqdm import tqdm


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 32 
PIN_MEMORY = True
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5 # 2.5e-5
WEIGHT_DECAY = 0.01 # original weight decay: 0.01
MAX_NORM = 1.0 # max gradient norm allowed
PCT_START = 0.1 # proportion of epoch spent warming up to max lr

# vit model parameters
VIT_IMAGE_PATCH = 16
VIT_FRAME_PATCH = 80
VIT_FRAMES = 640 # indicates current clip length being used
VIT_DIM = 512
VIT_DEPTH = 4
VIT_HEADS = 4
VIT_MLP_DIM = 768
VIT_DROPOUT = 0.1
VIT_EMB_DROPOUT = 0.1
VIT_POOL = 'mean'

wandb.init(
    project="emotion-recognition",
    name=f"pretrain_and_test",
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": MAX_NORM,
        "vit_dim": VIT_DIM,
        "vit_depth": VIT_DEPTH,
        "vit_heads": VIT_HEADS,
        "vit_mlp_dim": VIT_MLP_DIM,
        "vit_dropout": VIT_DROPOUT,
        "vit_emb_dropout": VIT_EMB_DROPOUT,
        "vit_pool": VIT_POOL,
        "vit_image_patch": VIT_IMAGE_PATCH,
        "vit_frame_patch": VIT_FRAME_PATCH,
    }
)

# helper method for getting all relevant classification metrics from predictions and targets
def get_metrics(y_true, y_pred, y_probs=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_probs is not None:
        try:
            if np.ndim(y_probs) == 1:
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

# dataset object for reading from h5 files as needed
class h5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None # lazy open file in __getitem__
        
    def __len__(self):
        # open file in a temporary context to get length if not already opened
        if self.file is None:
            with h5py.File(self.file_path, 'r') as f:
                return f['samples'].shape[0]
        return self.file['samples'].shape[0]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            self.samples = self.file['samples']
            self.labels = self.file['labels']
        sample = self.samples[index]
        sample = np.expand_dims(sample, axis=0) # shape: (1, frames, height, width)
        label = self.labels[index]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __del__(self):
        if self.file is not None:
            self.file.close()


vit = ViT( # vision transformer parameters as suggested by Awan et al. 2024
    image_size=32,
    frames=VIT_FRAMES,
    image_patch_size=VIT_IMAGE_PATCH,
    frame_patch_size=VIT_FRAME_PATCH,
    num_classes=4,
    dim=VIT_DIM,
    depth=VIT_DEPTH,
    heads=VIT_HEADS,
    mlp_dim=VIT_MLP_DIM,
    channels=1,
    dropout=VIT_DROPOUT,
    emb_dropout=VIT_EMB_DROPOUT,
    pool=VIT_POOL,
).to(device)


# creating datasets/dataloaders
print("Loading training data...")
train_dataset = h5Dataset("data/deap_train_data.h5")
test_dataset = h5Dataset("data/deap_test_data.h5")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print("Training data initialized.")
print(f"Training set size: {len(train_dataset)} samples")


loss_fn = nn.CrossEntropyLoss() # cross entropy is appropriate for classification
optimizer = optim.AdamW(vit.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE,
    total_steps=len(train_loader)*NUM_EPOCHS,
    pct_start=PCT_START,
    div_factor=10
)

# record per-epoch train/validation losses
train_epoch_losses = []
val_epoch_losses = []
# training loop
print("\nTraining...")
for epoch in range(NUM_EPOCHS):
    vit.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", total=len(train_loader))
    for inputs, targets in progress_bar:
        for inputs, labels in train_loader:
            print(inputs.shape)  # Expected: (batch_size, channels, frames, height, width)
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = vit(inputs)
            loss = loss_fn(outputs, targets)
        # calculate loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # get gradient norms
        total_norm = 0.0
        for p in vit.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() **2
        total_norm = total_norm ** 0.5
        # print current epoch metrics
        current_lr = optimizer.param_groups[0]['lr']
        # update progress bar
        progress_bar.set_postfix({
            "LR": f"{current_lr:.2e}",
            "Loss": f"{loss:.4f}",
            "Grad Norm": f"{total_norm:.4f}"
        })
        # update with clipped norms
        torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * inputs.size(0)  # Accumulate loss over batches
    
    epoch_train_loss = running_loss / len(train_dataset)
    train_epoch_losses.append(epoch_train_loss) 
        
    # validation loop for each epoch
    print("\n\tValidation:")
    vit.eval()
    epoch_val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = vit(inputs)
                loss_val = loss_fn(outputs, targets)
            epoch_val_loss += loss_val.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            correct_val += torch.sum(preds == targets).sum().item()
            total_val += targets.size(0)
    # get and print metrics from validation
    epoch_val_loss /= total_val
    val_epoch_losses.append(epoch_val_loss)
    val_acc, val_prec, val_rec, val_f1, val_roc_auc = get_metrics(all_targets, all_preds, all_probs)
    roc_auc_str = f"{val_roc_auc:.4f}" if val_roc_auc is not None else "N/A"
    print(f"\tLoss: {epoch_val_loss:.4f} | Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | ROC AUC: {roc_auc_str}")
    print("\t:.......................................................................................................:\n")
    wandb.log({
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
print("\nFinished Training.")
del train_loader, train_dataset
gc.collect()

# testing loop
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

# calculate overall/valence-arousal metrics
print("Computing metrics...")
true_valence = (all_targets >= 2).astype(int)  # 0=low (0,1), 1=high (2,3)
pred_valence = (all_preds >= 2).astype(int)
valence_prob = all_probs[:, 2] + all_probs[:, 3]

# Arousal: even vs odd (0/2 vs 1/3)
true_arousal = (all_targets % 2).astype(int)  # 0=even (0,2), 1=odd (1,3)
pred_arousal = (all_preds % 2).astype(int)
arousal_prob = all_probs[:, 1] + all_probs[:, 3]

# get metrics for valence, arousal, and overall
valence_accuracy, valence_precision, valence_recall, valence_f1, valence_roc_auc = get_metrics(true_valence, pred_valence, valence_prob)
arousal_accuracy, arousal_precision, arousal_recall, arousal_f1, arousal_roc_auc = get_metrics(true_arousal, pred_arousal, arousal_prob)
overall_accuracy, overall_precision, overall_recall, overall_f1, overall_roc_auc = get_metrics(all_targets, all_preds, all_probs)

# get final confusion matrix
cm = confusion_matrix(all_targets, all_preds)

# log final metrics
wandb.log({
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

# save model
torch.save(vit.state_dict(), f"models/pretrained_and_tested.pth")

# creating confusion matrix image
class_names = ["LVLA", "LVHA", "HVLA", "HVHA"]
cm_df = pd.DataFrame(cm)
cm_df.to_csv(f"cm_df_pretrained_and_tested.csv", index=False)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()