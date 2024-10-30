from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
from abc import ABC
import torch

@dataclass
class BaseConfig(ABC):
    """
    Base Configuration Class that models should inherit from
    """
    model_name: str
    max_length: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int = 41
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent = 2)

    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
    