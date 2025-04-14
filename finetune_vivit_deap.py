import numpy as np
import torch
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import VivitForVideoClassification, VivitImageProcessor, VivitConfig, Trainer, TrainingArguments, EarlyStoppingCallback
import torchvision.transforms.v2 as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_WORKERS = 16
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NOISE_PROB = 0.0
TRAIN_PROB = 0.8
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_CLASSES = 3
NUM_UNFROZEN_LAYERS = 2 # number of transformer layers that get unfrozen for training alongside mlp head

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

data = np.load("all_deap_cwt_data.npz")
all_samples = data['samples']
all_labels = data['labels']

train_samples, test_samples, train_labels, test_labels = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

train_dataset = CWTDataset(train_samples, train_labels, preprocess_transform)
eval_dataset = CWTDataset(test_samples, test_labels, preprocess_transform)

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
#model = torch.compile(model, mode="reduce-overhead") # compile model to speed up training

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

training_args = TrainingArguments(
    output_dir=f"models/vivit_finetuned-{NUM_UNFROZEN_LAYERS}-layers_deap",
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
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()
final_model_path = f"models/vivit_finetuned-{NUM_UNFROZEN_LAYERS}-layers_deap/best_{NUM_UNFROZEN_LAYERS}-layers"
trainer.save_model(final_model_path)
image_processor.save_pretrained(final_model_path)
eval_results = trainer.evaluate(eval_dataset)
print("Test results:\n", eval_results)
# get final confusion matrix
predictions_output = trainer.predict(eval_dataset)
y_pred = predictions_output.predictions.argmax(axis=-1)
y_true = predictions_output.label_ids
class_names = ['Unpleasant', 'Neutral', 'Pleasant']
cm = confusion_matrix(y_true, y_pred)
print("\nFinal Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.savefig(f"{final_model_path}/confusion_matrix.png")