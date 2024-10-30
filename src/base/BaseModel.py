from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
import pandas as pd
from config import BaseConfig


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders."""
        pass

    @abstractmethod
    def train_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any
    ) -> float:
        """Train for one epoch."""
        pass

    @abstractmethod
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        pass

    @abstractmethod
    def predict(self, text: str) -> Any:
        """Make a prediction for a single text input."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model state."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, config: BaseConfig) -> "BaseModel":
        """Load model state."""
        pass
