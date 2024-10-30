from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, PreTrainedTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataclasses import dataclass

# get base model and base config
from ..src.base.config import BaseConfig
from ..src.base.model import BaseModel

@dataclass
class TransformerConfig(BaseConfig):
    """
    Configuration for the transformer-based model
    """
    # Model specific
    model_name: str = "distilbert-based-uncased"
    max_length: int = 128
    dropout_rate: float = 0.25

    # Training specific
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0

    # Data specific
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    text_column: str = 'text'
    label_column: str = 'label'

    # System Specific
    num_workers: int = 4
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class BankTransactionDataset(Dataset):
    """
    Used to initialize the data in a custom dataset
    """

    def __init__(self, texts, labels, tokenizer: PreTrainedTokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

class Transformer(BaseModel):
    """
    Transformer-based model implementation
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.device = torch.device(config.device)
        self.label_encoder = LabelEncoder()
        self._initialize_model()

    def _initialize_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            hidden_dropout_prob = self.config.dropout_rate
        )
        self.model.to(self.config.device)

    def prepare_data(self, df):
        encoded_labels = self.label_encoder.fit_transform(df[self.config.label_column])
        self.num_labels = len(encoded_labels.classes_)

        # Separate into train/test
        train_text, val_text, train_labels, val_labels = train_test_split(
            df[self.config.text_column].values,
            encoded_labels,
            text_size = self.config.val_ratio,
            random_state = self.config.random_seed,
            stratify = encoded_labels
        )

        # Create datasets
        train_dataset = BankTransactionDataset(
            train_text, 
            train_labels,
            self.tokenizer,
            self.config.max_length
        )
        val_dataset = BankTransactionDataset(
            val_text,
            val_labels,
            self.tokenizer,
            self.config.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.AdamW, scheduler) -> float:
        self.model.train()
        total_loss = 0

        # create progress bar from tqdm
        progress_bar = tqdm(train_loader, desc = "Training")
        for batch in progress_bar:
            # Put batch to device
            batch = {k: v.to(self.config.device) for k,v in batch.items()}

            # forward
            outputs = self.model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                labels = batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item()

            # backwards pass   
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val
            )

            optimizer.step()
            scheduler.step()

            # update progress
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate model
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc = "Evaluating"):
                # move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # forward pass
                outputs = self.model(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    labels = batch['labels']
                )

                total_loss += outputs.loss.item()

                # get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim = 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # calculate accuracy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_loss = total_loss / len(val_loader)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'primary_metric': accuracy
        }
    
    def predict(self, text: str):
        """Make a prediction for a single text input."""
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
        
        # Get predicted label and probabilities
        predicted_label = self.label_encoder.inverse_transform([pred_class])[0]
        class_probabilities = {
            self.label_encoder.inverse_transform([i])[0]: prob.item()
            for i, prob in enumerate(probs[0])
        }
        
        return {
            'predicted_label': predicted_label,
            'confidence': probs[0][pred_class].item(),
            'class_probabilities': class_probabilities
        }

    def save_model(self, path: str) -> None:
        """Save model state and configuration."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'config': self.config
        }, path)
    
    @classmethod
    def load_model(cls, path: str, config):
        """Load model state and configuration."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Use loaded config if none provided
        if config is None:
            config = checkpoint['config']
        
        # Initialize model
        model = cls(config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.tokenizer = checkpoint['tokenizer']
        model.label_encoder = checkpoint['label_encoder']
        
        return model

if __name__ == "__main__":
    # Create configuration
    config = TransformerConfig(
        model_name='distilbert-base-uncased',
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        text_column='memo',
        label_column='category'
    )
    
    # Initialize model
    model = Transformer(config)
    
    # Load sample data
    df = pd.DataFrame({
        'memo': ['sample text 1', 'sample text 2'],
        'category': ['class1', 'class2']
    })
    
    # Prepare data
    train_loader, val_loader = model.prepare_data(df)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train and evaluate
    for epoch in range(config.num_epochs):
        train_loss = model.train_epoch(train_loader, optimizer, scheduler)
        metrics = model.evaluate(val_loader)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    # Make prediction
    result = model.predict("sample prediction text")
    print(f"Prediction: {result['predicted_label']} (Confidence: {result['confidence']:.4f})")



