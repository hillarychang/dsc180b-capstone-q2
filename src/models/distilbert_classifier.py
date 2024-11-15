import sys
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def distilbert_function(train_file, test_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and test data
    train_data = []
    train_labels = []
    with open(train_file, 'r') as f:
        for line in f:
            label, text = line.strip().split(' ', 1)
            train_labels.append(label.replace("__label__", ""))
            train_data.append(text)
    
    test_data = []
    test_labels = []
    with open(test_file, 'r') as f:
        for line in f:
            label, text = line.strip().split(' ', 1)
            test_labels.append(label.replace("__label__", ""))
            test_data.append(text)

    print("Data loaded successfully")

    # Encode labels
    unique_labels = list(set(train_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    train_labels_encoded = [label_to_index[label] for label in train_labels]
    test_labels_encoded = [label_to_index[label] for label in test_labels]

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(unique_labels)
    )
    model.to(device)

    # Create Dataset objects
    train_dataset = TextDataset(train_data, train_labels_encoded, tokenizer, max_len=128)
    test_dataset = TextDataset(test_data, test_labels_encoded, tokenizer, max_len=128)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 2
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    print("Training complete")

    # Evaluation loop
    print("Starting evaluation...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Decoding predictions and true labels
    predictions_decoded = [index_to_label[pred] for pred in predictions]
    true_labels_decoded = [index_to_label[label] for label in true_labels]

    # Evaluating predictions
    accuracy = accuracy_score(true_labels_decoded, predictions_decoded)
    print("\nAccuracy:", accuracy)

    # Generate classification report
    class_report = classification_report(true_labels_decoded, predictions_decoded, target_names=unique_labels)
    print("\nClassification Report:\n", class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels_decoded, predictions_decoded, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
    print("\nConfusion Matrix:\n", conf_matrix_df)

if __name__ == "__main__":
    # Use same dataset files as FastText
    train_file = 'train_data.txt'
    test_file = 'test_data.txt'
    distilbert_function(train_file, test_file)