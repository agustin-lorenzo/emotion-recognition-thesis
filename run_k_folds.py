import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import VivitForVideoClassification, VivitImageProcessor, Trainer, TrainingArguments, EarlyStoppingCallback
import torchvision.transforms.v2 as transforms
import csv
import wandb
import os
import re
import copy
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initalize constants
BATCH_SIZE = 32
NUM_WORKERS = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5 # 3e-5 performs best
WEIGHT_DECAY = 0.01
NOISE_PROB = 0.0
TRAIN_PROB = 0.8
MODEL_NAME_HF = "google/vivit-b-16x2-kinetics400"
NUM_CLASSES = 3

# get config arguments from flags
parser = argparse.ArgumentParser(
    description="Run ViViT fine‑tuning; single‑phase (D0/P2) or two‑phase (D0P1/P2D0)."
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Config: 'D' for DEAP or 'P' for Private, followed by the number of unfrozen layers from 0-2 (e.g., D0, P2, D1, etc.).\n        A 'chained' flag (D#P#) can be used to start the second finetuning phase with a previously trained model"
)
args = parser.parse_args()
config = args.config.upper()

# check finetuning scenario
m = re.fullmatch(r"([DP])([0-2])(?:([DP])([0-2]))?", config)
if not m:
    raise ValueError("`--config` must be D# or P# or D#P# or P#D#, with # equal to 0, 1, or 2")
phase1_ds, phase1_freeze, phase2_ds, phase2_freeze = m.groups()
phase1_freeze = int(phase1_freeze)
phase2_freeze = int(phase2_freeze) if phase2_ds else None

if phase2_ds:
    # further training with model previously finetuned on DEAP dataset
    phase1_tag = f"{phase1_ds}{phase1_freeze}"
    checkpoint = f"models/{phase1_tag}"
    print(f"loading pretrained checkpoint from phase1: {checkpoint}")
    base_model = VivitForVideoClassification.from_pretrained(checkpoint)
    DATASET = ("data/all_deap_cwt_data.npz" if phase2_ds == "D" else "data/all_private_cwt_data.npz")
    num_unfrozen = phase2_freeze
else:
    # finetuning vanilla hugging face model on private data directly
    print(f"loading base ViViT from hugging face: {MODEL_NAME_HF}")
    base_model = VivitForVideoClassification.from_pretrained(MODEL_NAME_HF)
    DATASET = ("data/all_deap_cwt_data.npz" if phase1_ds == "D" else "data/all_private_cwt_data.npz")
    num_unfrozen = phase1_freeze
    num_features = base_model.classifier.in_features
    base_model.classifier = torch.nn.Linear(num_features, NUM_CLASSES)
    base_model.config.num_labels = NUM_CLASSES
    base_model.num_labels = NUM_CLASSES
    
# load model and freeze backbone to only train mlp head
for p in base_model.vivit.parameters():
    p.requires_grad = False
# unfreeze given number of transformer layers from flag
print(f"unfreezing {num_unfrozen} transformer layers...")
for i in range(num_unfrozen):
    for p in base_model.vivit.encoder.layer[-1 - i].parameters():
        p.requires_grad = True

out_tag = config
base_models_dir = f"models/{out_tag}"
base_metrics_dir = f"metrics/{out_tag}"
csv_results_path = f"{base_metrics_dir}/cv_results.csv"
os.makedirs(base_models_dir, exist_ok=True)
os.makedirs(base_metrics_dir, exist_ok=True)

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

# using a wraper for Huggingface's Trainer to account for class imbalance
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

image_processor = VivitImageProcessor.from_pretrained(MODEL_NAME_HF)
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

all_fold_accuracies = []
all_fold_recalls = []
all_fold_precisions = []
all_fold_aucs = []
all_fold_f1s = []
overall_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
with open(csv_results_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ["fold", "loss", "accuracy", "recall", "precision", "roc_auc", "f1"]
    writer.writerow(header)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_idx, test_idx in skf.split(all_samples, all_labels):
        wandb.init(
            project="emotion-recognition",
            name=f"{out_tag}-stride2_fold-{fold}",
            config={
                "fold": fold,
                "model_name": MODEL_NAME_HF,
                "num_unfrozen_layers": num_unfrozen,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
            },
            reinit=True
        )
        
        train_samples, train_labels = all_samples[train_idx], all_labels[train_idx]
        test_samples, test_labels = all_samples[test_idx], all_labels[test_idx]
        train_dataset = CWTDataset(train_samples, train_labels, preprocess_transform)
        eval_dataset = CWTDataset(test_samples, test_labels, preprocess_transform)
        
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"fold {fold} class weights: {class_weights}")

        # make a copy of the base model for the fold
        model = copy.deepcopy(base_model)

        training_args = TrainingArguments(
            output_dir=f"{base_models_dir}/fold_{fold}",
            eval_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_train_epochs=NUM_EPOCHS,
            logging_steps=max(1, len(train_dataset) // (BATCH_SIZE * 4)),
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_strategy="epoch",
            save_total_limit=1,
            push_to_hub=False,
            report_to="wandb"
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        trainer.train()
        
        final_model_path = f"{base_models_dir}/best_fold_{fold}"
        trainer.save_model(final_model_path)
        image_processor.save_pretrained(final_model_path)
        
        fold_dir = f"{base_metrics_dir}/fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        eval_results = trainer.evaluate(eval_dataset)
        acc = eval_results.get('eval_accuracy', 0.0)
        rec = eval_results.get('eval_recall', 0.0)
        pre = eval_results.get('eval_precision', 0.0)
        auc = eval_results.get('eval_roc_auc', 0.0)
        f1 = eval_results.get('eval_f1', 0.0)
        loss = eval_results.get('eval_loss', 0.0)

        all_fold_accuracies.append(acc)
        all_fold_recalls.append(rec)
        all_fold_precisions.append(pre)
        all_fold_aucs.append(auc)
        all_fold_f1s.append(f1)
        row_data = [fold, f"{loss:.6f}", f"{acc:.6f}", f"{rec:.6f}", f"{pre:.6f}", f"{auc:.6f}", f"{f1:.6f}"]
        with open(f"{fold_dir}/metrics.txt","w") as ft:
            ft.write(f"loss={loss:.6f}\nacc={acc:.4f}\nrecall={rec:.4f}\nprec={pre:.4f}\nauc={auc:.4f}\nf1={f1:.4f}\n")
        writer.writerow(row_data)
        
        # get confusion matrix for fold
        predictions_output = trainer.predict(eval_dataset)
        y_pred = predictions_output.predictions.argmax(axis=-1)
        y_true = predictions_output.label_ids
        class_names = ['Unpleasant', 'Neutral', 'Pleasant']
        cm = confusion_matrix(y_true, y_pred)
        overall_cm += cm
        print("\nFinal Confusion Matrix:\n", cm)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.savefig(f"{fold_dir}/fold-{fold}_confusion_matix.png")
        plt.close()
        wandb.finish()
        fold += 1
    
mean_accuracy = np.mean(all_fold_accuracies)
std_accuracy = np.std(all_fold_accuracies)
mean_f1 = np.mean(all_fold_f1s)
std_f1 = np.std(all_fold_f1s)
mean_recall = np.mean(all_fold_recalls)
std_recall = np.std(all_fold_recalls)
mean_precision = np.mean(all_fold_precisions)
std_precision = np.std(all_fold_precisions)
mean_auc = np.mean(all_fold_aucs)
std_auc = np.std(all_fold_aucs)
np.savetxt(f"{base_metrics_dir}/overall_confusion_matrix.csv", overall_cm, fmt="%d", delimiter=",")

with open(f"{base_metrics_dir}/averages.txt", "w") as f:
    f.write("--- Cross-Validation Summary ---\n")
    f.write(f"Folds: {skf.get_n_splits()}\n")
    f.write(f"Average Accuracy:  {mean_accuracy:.4f} +/- {std_accuracy:.4f}\n")
    f.write(f"Average F1-Score:  {mean_f1:.4f} +/- {std_f1:.4f}\n")
    f.write(f"Average Recall:    {mean_recall:.4f} +/- {std_recall:.4f}\n")
    f.write(f"Average Precision: {mean_precision:.4f} +/- {std_precision:.4f}\n")
    f.write(f"Average ROC AUC:   {mean_auc:.4f} +/- {std_auc:.4f}\n\n")
    f.write("Individual Fold Accuracies: " + ", ".join(f"{x:.4f}" for x in all_fold_accuracies) + "\n\n")
    f.write("Raw Lists:\n")
    f.write(f"\taccuracies: {all_fold_accuracies}\n")
    f.write(f"\tprecisions: {all_fold_precisions}\n")
    f.write(f"\trecalls: {all_fold_recalls}\n")
    f.write(f"\troc aucs: {all_fold_aucs}\n")
    f.write(f"\tf1s: {all_fold_f1s}\n")

class_names = ['Unpleasant', 'Neutral', 'Pleasant']
disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=class_names)
disp.plot(cmap='viridis', values_format='d', colorbar=False)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title(out_tag)
plt.tight_layout()
plt.savefig(f"{base_metrics_dir}/overall_confusion_matrix.svg", format="svg")
plt.close()
