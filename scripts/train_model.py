#!/usr/bin/env python
"""
Training script for table extraction models.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableDataset(Dataset):
    """Dataset for table detection/recognition training."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing data
            split: train/val/test
            transform: Image transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load annotations
        annotation_file = self.data_dir / f"{split}_annotations.json"
        if annotation_file.exists():
            with open(annotation_file) as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
            logger.warning(f"No annotation file found: {annotation_file}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a sample."""
        ann = self.annotations[idx]
        
        # Load image
        from PIL import Image
        img_path = self.data_dir / "images" / ann["image"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Get targets
        target = {
            "boxes": torch.tensor(ann.get("boxes", []), dtype=torch.float32),
            "labels": torch.tensor(ann.get("labels", []), dtype=torch.int64),
        }
        
        return image, target


class TrainingConfig:
    """Training configuration."""
    
    def __init__(
        self,
        model_type: str = "detection",
        batch_size: int = 8,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        gradient_clip: float = 1.0,
        save_interval: int = 10,
        eval_interval: int = 5,
        early_stopping_patience: int = 20,
        mixed_precision: bool = True,
        num_workers: int = 4,
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.gradient_clip = gradient_clip
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        self.num_workers = num_workers


class Trainer:
    """Model trainer."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: Path = Path("outputs"),
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            self.val_loader = None
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.early_stopping_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self.model(images, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model(images, targets)
                loss.backward()
                
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple:
        """Validate model."""
        if not self.val_loader:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                loss = self.model(images, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics (implement specific metrics based on task)
        metric = avg_loss  # Placeholder
        
        return avg_loss, metric
    
    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
            "config": vars(self.config),
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.history = checkpoint["history"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Run full training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Config: {vars(self.config)}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_loss, metric = self.validate()
                
                if val_loss is not None:
                    self.history["val_loss"].append(val_loss)
                    self.history["val_metric"].append(metric)
                    logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f}")
                    
                    # Check for improvement
                    if metric < self.best_metric:
                        self.best_metric = metric
                        self.early_stopping_counter = 0
                        self.save_checkpoint("best_model.pt")
                    else:
                        self.early_stopping_counter += 1
                    
                    # Early stopping
                    if self.early_stopping_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
            
            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint()
            
            # Update scheduler
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        logger.info("Training completed")


def train_detection_model(args):
    """Train table detection model."""
    from src.detection.yolo_detector import YOLOTableDetector
    
    logger.info("Training detection model...")
    
    # Initialize model
    detector = YOLOTableDetector(model_path=args.pretrained)
    
    # Train using YOLO's built-in training
    detector.train(
        data_yaml=str(Path(args.data_dir) / "data.yaml"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


def train_structure_model(args):
    """Train structure recognition model."""
    from src.structure.transformer_recognizer import TableStructureModel
    
    logger.info("Training structure recognition model...")
    
    # Initialize model
    model = TableStructureModel()
    
    # Setup datasets
    train_dataset = TableDataset(
        data_dir=Path(args.data_dir),
        split="train",
    )
    
    val_dataset = TableDataset(
        data_dir=Path(args.data_dir),
        split="val",
    )
    
    # Training config
    config = TrainingConfig(
        model_type="structure",
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=Path(args.output_dir),
    )
    
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train table extraction models"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["detection", "structure"],
        required=True,
        help="Type of model to train",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Output directory for checkpoints and logs",
    )
    
    parser.add_argument(
        "--pretrained",
        type=str,
        help="Path to pre-trained model weights",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    # Train
    if args.model_type == "detection":
        train_detection_model(args)
    elif args.model_type == "structure":
        train_structure_model(args)


if __name__ == "__main__":
    main()
