"""
finetune_vivit.py
Author: Agustin Lorenzo

This is a script used for training a model (i.e. Google's ViViT) on an entire dataset without train/test splits or k-fold cross validation.
For my thesis, this is used for the first phase of finetuning on the public DEAP dataset (D0, D1, D2).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import VivitForVideoClassification, VivitImageProcessor, Trainer, TrainingArguments, EarlyStoppingCallback
import torchvision.transforms.v2 as transforms
import wandb
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set constants from CLI flags
parser = argparse.ArgumentParser(description="Run K-Fold CV for ViViT fine-tuning on either DEAP or Private dataset, with a specified number of unfrozen transformer layers.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Config string: 'D' for DEAP or 'P' for Private, followed by the number of unfrozen layers from 0-2 (e.g., D0, P2, D1, etc.)."
)
args = parser.parse_args()
config_str = args.config

if config_str[0] == 'D':
    DATASET = "data/all_deap_cwt_data.npz"
elif config_str[0] == 'P':
    DATASET = "data/all_private_cwt_data.npz"
else:
    raise ValueError("invalid dataset code.")
BATCH_SIZE = 32
NUM_WORKERS = 16
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5 # 3e-5 performs best
WEIGHT_DECAY = 0.01
NOISE_PROB = 0.0
TRAIN_PROB = 0.8
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_CLASSES = 3
NUM_UNFROZEN_LAYERS = int(config_str[1]) # number of transformer layers that get unfrozen for training alongside mlp head
                        # 2 performs best

FINETUNING_TYPE = config_str # type of finetuning:
                             # D - DEAP dataset
                             # P - Private dataset
                             # number following D/P - number of unfrozen transformer layers
                             # D#P# - previous finetuned model on DEAP finetuned again on private dataset
                                # TODO: add support for re-finetuning on private data
                                            
base_metrics_dir = f"metrics/{FINETUNING_TYPE}"
base_models_dir = f"models/{FINETUNING_TYPE}"

os.makedirs(base_metrics_dir, exist_ok=True)
os.makedirs(base_models_dir, exist_ok=True)

wandb.init(
    project="emotion-recognition",
    name=f"{FINETUNING_TYPE}-stride2",
    config={
        "model_name": MODEL_NAME,
        "num_unfrozen_layers": NUM_UNFROZEN_LAYERS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
    },
    reinit=True
)

# function for computing classification metrics: accuracy, precision, recall, f1, roc auc
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="macro", zero_division=0)
    rec = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    try:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        roc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception as e:
        roc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

# using a wrapper for Huggingface's Trainer to account for class imbalance
# uses class weights in loss function
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# loading data and preparing/transforming it for vivit input    
image_processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
image_size = (224, 224)
image_mean = image_processor.image_mean
image_std = image_processor.image_std
preprocess_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.unsqueeze(1)),
    transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
    transforms.Resize(size=image_size, antialias=False),
    transforms.Normalize(mean=image_mean, std=image_std),
])

class CWTDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip = torch.tensor(self.samples[idx], dtype=torch.float32)
        if self.transform:
            clip = self.transform(clip)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"pixel_values": clip, "labels": label}

data = np.load(DATASET)
all_samples = data['samples']
all_labels = data['labels']
dataset = CWTDataset(all_samples, all_labels, preprocess_transform)

class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# load model and freeze backbone to only train mlp head
model = VivitForVideoClassification.from_pretrained(MODEL_NAME)
for param in model.vivit.parameters():
    param.requires_grad = False
# unfreeze given number of transformer layers if desired
encoder_layers = model.vivit.encoder.layer
for i in range(NUM_UNFROZEN_LAYERS):
    for param in encoder_layers[11-i].parameters():
        param.requires_grad = True
# adjust classifier head for 3 classes
num_features = model.classifier.in_features 
model.classifier = torch.nn.Linear(num_features, 3) 
model.config.num_labels = 3
model.num_labels = 3
model.to(device)

# setting hyperparameters and beginning training
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=f"{base_models_dir}",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    #per_gpu_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=max(1, len(dataset) // (BATCH_SIZE * 4)),
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    push_to_hub=False,
    report_to="wandb"
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset, # using train data again for eval
                          # not actually evaluating model, just using it to determine best model from all epochs
                          # only applies to D0, D1, and D2
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)
trainer.train()

final_model_path = base_models_dir
trainer.save_model(final_model_path)
image_processor.save_pretrained(final_model_path)
wandb.finish()
