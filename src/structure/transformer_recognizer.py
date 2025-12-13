"""
Transformer-based table structure recognizer.
Inspired by TableFormer and similar architectures.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from .models import Cell, CellType, TableStructure, StructureRecognitionResult
from .recognizer import TableStructureRecognizer

logger = logging.getLogger(__name__)


class TableTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for table images.
    Extracts features from table images for structure recognition.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            num_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Encoded features (B, num_patches + 1, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)
        
        return x


class CellDecoder(nn.Module):
    """
    Decoder for predicting cell positions and attributes.
    Uses attention over encoded features to predict cells.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        max_cells: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_cells = max_cells
        
        # Cell queries
        self.cell_queries = nn.Parameter(
            torch.randn(1, max_cells, embed_dim) * 0.02
        )
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output heads
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # x1, y1, x2, y2 normalized
            nn.Sigmoid()
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # cell / no-cell
        )
        
        self.span_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # row, col, rowspan, colspan
        )
        
        self.type_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(CellType))  # cell type classification
        )
    
    def forward(
        self,
        encoder_output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_output: Encoder features (B, seq_len, embed_dim)
            
        Returns:
            Dictionary with predictions:
            - boxes: (B, max_cells, 4) normalized coordinates
            - classes: (B, max_cells, 2) cell/no-cell logits
            - spans: (B, max_cells, 4) row, col, rowspan, colspan
            - types: (B, max_cells, num_types) cell type logits
        """
        B = encoder_output.shape[0]
        
        # Expand queries for batch
        queries = self.cell_queries.expand(B, -1, -1)
        
        # Decode
        decoded = self.decoder(queries, encoder_output)
        
        # Predict outputs
        return {
            "boxes": self.bbox_head(decoded),
            "classes": self.class_head(decoded),
            "spans": self.span_head(decoded),
            "types": self.type_head(decoded)
        }


class TableStructureModel(nn.Module):
    """
    Complete table structure recognition model.
    Combines encoder and decoder for end-to-end prediction.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_heads: int = 12,
        encoder_layers: int = 12,
        decoder_heads: int = 8,
        decoder_layers: int = 6,
        max_cells: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = TableTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        self.decoder = CellDecoder(
            embed_dim=embed_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            max_cells=max_cells,
            dropout=dropout
        )
        
        self.image_size = image_size
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Dictionary with cell predictions
        """
        features = self.encoder(images)
        outputs = self.decoder(features)
        return outputs


class TransformerTableRecognizer(TableStructureRecognizer):
    """
    Transformer-based table structure recognizer.
    
    Example:
        recognizer = TransformerTableRecognizer(model_path="weights/structure.pt")
        result = recognizer.recognize(table_image)
        print(f"Found {len(result.table.cells)} cells")
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        image_size: int = 512,
        max_cells: int = 500
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold
        )
        self.image_size = image_size
        self.max_cells = max_cells
    
    def load_model(self) -> None:
        """Load transformer model."""
        self.model = TableStructureModel(
            image_size=self.image_size,
            max_cells=self.max_cells
        )
        
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading structure model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning("No model weights found, using random initialization")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info(f"Structure model loaded on device: {self.device}")
    
    def _preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Preprocess image for model input."""
        import cv2
        
        # Resize to model input size
        resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert BGR to RGB
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        elif len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def _recognize(
        self,
        image: NDArray[np.uint8],
        table_bbox: Optional[tuple] = None
    ) -> StructureRecognitionResult:
        """Run structure recognition."""
        start_time = time.time()
        
        # Get original image size for coordinate scaling
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Parse predictions
        cells = self._parse_predictions(outputs, orig_w, orig_h)
        
        inference_time = time.time() - start_time
        
        # Build table structure
        table = TableStructure(
            cells=cells,
            bbox=table_bbox,
            has_header=any(c.cell_type == CellType.HEADER for c in cells)
        )
        
        return StructureRecognitionResult(
            table=table,
            inference_time=inference_time,
            model_name="TransformerTableRecognizer"
        )
    
    def _parse_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        img_width: int,
        img_height: int
    ) -> List[Cell]:
        """Parse model outputs into Cell objects."""
        boxes = outputs["boxes"][0].cpu().numpy()  # (max_cells, 4)
        classes = outputs["classes"][0].cpu().numpy()  # (max_cells, 2)
        spans = outputs["spans"][0].cpu().numpy()  # (max_cells, 4)
        types = outputs["types"][0].cpu().numpy()  # (max_cells, num_types)
        
        cells = []
        
        # Apply softmax to get probabilities
        class_probs = self._softmax(classes)
        type_probs = self._softmax(types)
        
        for i in range(len(boxes)):
            # Check if this is a valid cell
            cell_prob = class_probs[i, 1]  # Probability of being a cell
            
            if cell_prob < self.confidence_threshold:
                continue
            
            # Get bounding box (normalized to absolute coordinates)
            x1, y1, x2, y2 = boxes[i]
            bbox = (
                x1 * img_width,
                y1 * img_height,
                x2 * img_width,
                y2 * img_height
            )
            
            # Get row/col/span info
            row = max(0, int(round(spans[i, 0])))
            col = max(0, int(round(spans[i, 1])))
            rowspan = max(1, int(round(spans[i, 2])))
            colspan = max(1, int(round(spans[i, 3])))
            
            # Get cell type
            type_idx = np.argmax(type_probs[i])
            cell_type = list(CellType)[type_idx]
            
            cells.append(Cell(
                row=row,
                col=col,
                rowspan=rowspan,
                colspan=colspan,
                bbox=bbox,
                confidence=float(cell_prob),
                cell_type=cell_type
            ))
        
        return cells
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        output_dir: str = "checkpoints"
    ):
        """
        Train the structure recognition model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Directory to save checkpoints
        """
        if not self._is_loaded:
            self.load_model()
        
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_dataloader:
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    images = batch["images"].to(self.device)
                    targets = batch["targets"]
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, targets)
                    val_loss += loss.item()
            
            scheduler.step()
            
            avg_train_loss = train_loss / len(train_dataloader)
            avg_val_loss = val_loss / len(val_dataloader)
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(),
                    output_path / "best_model.pt"
                )
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            output_path / "final_model.pt"
        )
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute training loss."""
        # Box regression loss (L1)
        box_loss = nn.functional.l1_loss(
            outputs["boxes"],
            targets["boxes"]
        )
        
        # Classification loss (cross entropy)
        class_loss = nn.functional.cross_entropy(
            outputs["classes"].view(-1, 2),
            targets["classes"].view(-1).long()
        )
        
        # Span regression loss
        span_loss = nn.functional.l1_loss(
            outputs["spans"],
            targets["spans"]
        )
        
        # Type classification loss
        type_loss = nn.functional.cross_entropy(
            outputs["types"].view(-1, len(CellType)),
            targets["types"].view(-1).long()
        )
        
        # Weighted sum
        total_loss = box_loss + class_loss + span_loss + 0.5 * type_loss
        
        return total_loss
