import numpy as np
import torch
import random
from torch.utils.data import Dataset
import h5py
from run_k_folds import add_gaussian_noise
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from transformers import VivitForVideoClassification, Trainer, TrainingArguments

BATCH_SIZE = 32
NUM_EPOCHS = 50
NOISE_PROB = 0.0

# dataset object for reading from h5 files as needed
class h5Dataset(Dataset):
    def __init__(self, file_path, training=False):
        self.training = training
        self.file_path = file_path
        # just get metadata when oopening
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = f['samples'].shape[0]
        self._file = None
        self._samples = None
        self._labels = None
        
    def __len__(self):
        return self.num_samples
    
    def _open_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, 'r')
            self._samples = self._file['samples']
            self._labels = self._file['labels']
    
    def __getitem__(self, index):
        self._open_file()
        sample = self._samples[index]
        sample = np.expand_dims(sample, axis=0) # shape: (1, frames, height, width)
        sample = np.repeat(sample, 3, axis=0)   # shape: (3, frames, height, width) for pseudo-rgb
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        sample_tensor = sample_tensor / 255.0
        mean = sample_tensor.mean()
        std = sample_tensor.std()
        epsilon = 1e-6
        sample_tensor = (sample_tensor - mean) / (std + epsilon)
        if self.training and random.random() < NOISE_PROB:
            # add extra noise to sample if training
            sample_np = sample_tensor.numpy()
            noisy_sample = add_gaussian_noise(sample_np, snr_db=6)
            sample_tensor = torch.tensor(noisy_sample, dtype=torch.float32)
        label = self._labels[index]
        label = torch.tensor(label, dtype=torch.long)
        
        return sample_tensor, label
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._samples = None
            self._labels = None
    
    def __del__(self):
        # As a backup, also close in the destructor
        self.close()
        
    def __enter__(self):
        self._open_file()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ViViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, **inputs):
        return self.model(**inputs, interpolate_pos_encoding=True)

base_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
base_model.config.num_labels = 3
base_model.classifier = torch.nn.Linear(base_model.config.hidden_size, 3)
model = ViViTWrapper(base_model)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="macro", zero_division=0)
    rec = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    try:
        roc = roc_auc_score(labels, logits, multi_class="ovr", average="macro")
    except Exception as e:
        roc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

training_args = TrainingArguments(
    ouput_dir="models",
    eval_strategy="epoch",
    per_device_eval_batch_size=BATCH_SIZE,
    per_gpu_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=NUM_EPOCHS,
    save_strategy="epoch",
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    training_args=training_args,
    train_dataset=train_datset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  
)

trainer.train()

