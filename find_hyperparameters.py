import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
from vit_pytorch.vit_3d import ViT
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

# get absolute path so ray can find it from its temp working directory
data_dir = os.path.abspath("data")

# altered metrics function, only looking at accuracy
def get_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)
    return accuracy

# dataset class from training scripts
class h5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None  # lazy open file in __getitem__
        
    def __len__(self):
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
        sample = np.expand_dims(sample, axis=0)  # shape: (1, frames, height, width)
        label = self.labels[index]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def __del__(self):
        if self.file is not None:
            self.file.close()

# training function to be used by ray 
def train_model(config, checkpoint_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SIZE = 32
    train_dataset = h5Dataset(os.path.join(data_dir, "deap_train_data.h5"))
    test_dataset = h5Dataset(os.path.join(data_dir, "deap_test_data.h5"))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    # vit model build from hyperparameter search
    model = ViT(
        image_size=32,
        frames=640,
        image_patch_size=16,
        frame_patch_size=80,
        num_classes=4,
        dim=int(config["vit_dim"]),
        depth=int(config["vit_depth"]),
        heads=int(config["vit_heads"]),
        mlp_dim=int(config["vit_mlp_dim_factor"]) * int(config["vit_dim"]),
        channels=1,
        dropout=config["vit_dropout"],
        emb_dropout=config["vit_emb_dropout"],
        pool=config["vit_pool"]
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=config["pct_start"]
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # training loop
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            progress_bar.set_postfix({
                "Loss": f"{loss:.4f}",
            })
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)
        
        # validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss_val = loss_fn(outputs, targets)
                val_loss += loss_val.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        val_loss /= len(test_dataset)
        val_accuracy = get_metrics(all_targets, all_preds)
        # report metrics to Ray Tune
        tune.report({"loss": val_loss, "accuracy": val_accuracy})

# parameters space to be searched by ray
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-4),
    "weight_decay": tune.uniform(0.0, 0.1),
    "max_norm": tune.uniform(0.5, 5.0),
    "pct_start": tune.uniform(0.05, 0.3),
    "vit_dim": tune.choice([512, 768, 1024]),
    "vit_depth": tune.choice([4, 6, 8, 12]),
    "vit_heads": tune.choice([4, 8, 16]),
    "vit_mlp_dim_factor": tune.choice([2, 4]),
    "vit_dropout": tune.uniform(0.1, 0.5),
    "vit_emb_dropout": tune.uniform(0.1, 0.3),
    "vit_pool": tune.choice(['mean', 'cls']),
    "epochs": 10  # For tuning, run for 10 epochs
}

# ray poputlation based training scheduler
pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=1,  # perturb every iteration/epoch
    hyperparam_mutations={
        "learning_rate": tune.loguniform(1e-5, 1e-4),
        "weight_decay": tune.uniform(0.0, 0.1),
        "max_norm": tune.uniform(0.5, 5.0),
        "pct_start": tune.uniform(0.05, 0.3),
        "vit_dim": [512, 768, 1024],
        "vit_depth": [4, 6, 8, 12],
        "vit_heads": [4, 8, 16],
        "vit_mlp_dim_factor": [2, 4],
        "vit_dropout": tune.uniform(0.1, 0.5),
        "vit_emb_dropout": tune.uniform(0.1, 0.3),
        "vit_pool": ["mean", "cls"]
    }
)

reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

result = tune.run(
    train_model,
    resources_per_trial={"cpu": 8, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=pbt,
    progress_reporter=reporter,
    metric="loss", # optimize by minimizing loss
    mode="min",
    storage_path="file://" + os.path.abspath("ray_results")
)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: ", best_trial.config)
print("Best trial final validation loss: ", best_trial.last_result["loss"])
print("Best trial final validation accuracy: ", best_trial.last_result["accuracy"])
