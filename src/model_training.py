import argparse
from pathlib import Path
import yaml
import importlib
from typing import Type
import logging
from base.config import BaseConfig
from base.model import BaseModel
import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup
from text_cleaner import clean_text


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> BaseConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    model_type = config_dict["model_type"]
    config_class = get_config_class(model_type)
    return config_class(**config_dict)


def get_model_class(model_type: str, model_version) -> Type[BaseModel]:
    """Dynamically import and return the model class."""
    try:
        module = importlib.import_module(f"src.models.{model_type.lower()}")
        return getattr(module, model_version)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Model type {model_type} not found: {e}")


def get_config_class(model_type: str) -> Type[BaseConfig]:
    """Dynamically import and return the config class."""
    try:
        module = importlib.import_module(f"src.models.{model_type.lower()}")
        return getattr(module, f"{model_type}Config")
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Config for model type {model_type} not found: {e}")


def train_model(model: BaseModel, data: pd.DataFrame, output_dir: Path):
    """Train the model and save artifacts."""
    logger = logging.getLogger("train_model")

    # Prepare data
    train_loader, val_loader = model.prepare_data(data)

    learning_rate = float(model.config.learning_rate)
    weight_decay = float(model.config.weight_decay)
    warmup_ratio = float(model.config.warmup_ratio)

    total_steps = len(train_loader) * model.config.num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_metric = float("-inf")
    for epoch in range(model.config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{model.config.num_epochs}")

        # Train
        train_loss = model.train_epoch(train_loader, optimizer, scheduler)
        logger.info(f"Training loss: {train_loss:.4f}")

        # Evaluate
        metrics = model.evaluate(val_loader)
        logger.info(f"Validation metrics: {metrics}")

        # Save best model
        if metrics["primary_metric"] > best_metric:
            best_metric = metrics["primary_metric"]
            model.save_model(output_dir / "best_model.pt")
            model.config.save(output_dir / "config.json")
            logger.info(f"Saved new best model with metric: {best_metric:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("main")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration: {config}")

    # Load data
    data = clean_text(pd.read_parquet(args.data))
    logger.info(f"Loaded {len(data)} training examples")

    # Initialize model
    model_class = get_model_class(config.model_type, config.model_version)
    model = model_class(config)
    logger.info(f"Initialized model: {model.__class__.__name__}")

    # train model
    train_model(model, data, output_dir)


if __name__ == "__main__":
    main()
