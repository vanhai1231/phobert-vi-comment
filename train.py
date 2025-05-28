from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import load_and_preprocess_dataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Bước 1: Load và xử lý dữ liệu
dataset_path = "./data/social_comments.csv"
dataset, label_encoder = load_and_preprocess_dataset(dataset_path)

# Bước 2: Load tokenizer và model
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["comment"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Bước 3: Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Bước 4: Định nghĩa metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# Bước 5: TrainingArguments
training_args = TrainingArguments(
    output_dir="./phobert-4class",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

# Bước 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("test", tokenized_dataset["train"].select(range(100))),  # fallback
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Bước 7: Train
trainer.train()

# Bước 8: Lưu model
model.save_pretrained("./phobert-4class")
tokenizer.save_pretrained("./phobert-4class")
