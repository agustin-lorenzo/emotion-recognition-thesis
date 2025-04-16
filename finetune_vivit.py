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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
csv_results_path = f"{base_metrics_dir}/cv_results.csv"

os.makedirs(base_metrics_dir, exist_ok=True)
os.makedirs(base_models_dir, exist_ok=True)

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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

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

all_fold_accuracies = []
all_fold_recalls = []
all_fold_precisions = []
all_fold_aucs = []
all_fold_f1s = []
with open(csv_results_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ["fold", "loss", "accuracy", "recall", "precision", "roc_auc", "f1"]
    writer.writerow(header)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_idx, test_idx in skf.split(all_samples, all_labels):
        wandb.init(
            project="emotion-recognition",
            name=f"{FINETUNING_TYPE}_fold-{fold}",
            config={
                "fold": fold,
                "model_name": MODEL_NAME,
                "num_unfrozen_layers": NUM_UNFROZEN_LAYERS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
            },
            reinit=True
        )
        
        fold += 1
        train_samples, train_labels = all_samples[train_idx], all_labels[train_idx]
        test_samples, test_labels = all_samples[test_idx], all_labels[test_idx]
        train_dataset = CWTDataset(train_samples, train_labels, preprocess_transform)
        eval_dataset = CWTDataset(test_samples, test_labels, preprocess_transform)
        
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"fold {fold} class weights: {class_weights}")

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

        training_args = TrainingArguments(
            output_dir=f"{base_models_dir}/fold_{fold}",
            eval_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            per_gpu_eval_batch_size=BATCH_SIZE,
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
        # TODO: save metrics to .txt file in fold_dir as well
        writer.writerow(row_data)
        
        # get final confusion matrix
        predictions_output = trainer.predict(eval_dataset)
        y_pred = predictions_output.predictions.argmax(axis=-1)
        y_true = predictions_output.label_ids
        class_names = ['Unpleasant', 'Neutral', 'Pleasant']
        cm = confusion_matrix(y_true, y_pred)
        print("\nFinal Confusion Matrix:\n", cm)
        
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.savefig(f"{fold_dir}/fold-{fold}_confusion_matix.png")
        plt.close()
        wandb.finish()
    
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

# TODO: save these final metrics to base_metrics_dir as well
print("\n--- Cross-Validation Summary ---")
print(f"Folds: {skf.get_n_splits()}")
print(f"Average Accuracy:  {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
print(f"Average F1-Score:  {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"Average Recall:    {mean_recall:.4f} +/- {std_recall:.4f}")
print(f"Average Precision: {mean_precision:.4f} +/- {std_precision:.4f}")
print(f"Average ROC AUC:   {mean_auc:.4f} +/- {std_auc:.4f}")
print("\nIndividual Fold Accuracies:", [f"{acc:.4f}" for acc in all_fold_accuracies])
print("---------------------------------")

print("\n\nRaw Lists:\n")
print("\taccuracies: ", all_fold_accuracies)
print("\tprecisions: ", all_fold_precisions)
print("\trecalls: ", all_fold_recalls)
print("\troc aucs: ", all_fold_aucs)
print("\tf1s: ", all_fold_f1s)