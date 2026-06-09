import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd


class SmartProductDataset(Dataset):
    """Custom Dataset designed to yield Review Text, Image Tensors, and Target Ratings."""
    def __init__(self, reviews_parquet_path, meta_parquet_path, asin_split_csv, image_dir='data/images', transform=None):
        self.reviews = pd.read_parquet(reviews_parquet_path)
        self.meta = pd.read_parquet(meta_parquet_path).set_index('parent_asin')
        self.allowed_asins = set(pd.read_csv(asin_split_csv).squeeze().tolist())
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Filter dataset to match specific train/val/test split indices
        self.reviews = self.reviews[self.reviews['parent_asin'].isin(self.allowed_asins)].reset_index(drop=True)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        row = self.reviews.iloc[idx]
        asin = row['parent_asin']
        review_text = str(row['text'])
        rating = torch.tensor(row['rating'] - 1, dtype=torch.long)  # Map 1-5 ratings scale to 0-4 idx target

        # Resolve Multimodal Product Image File Path
        img_path = self.image_dir / f"{asin}.jpg"
        if img_path.exists():
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback placeholder tensor if product didn't have an available/successful image download
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)
        else:
            # Fallback basic conversion to tensor structure if transform object omitted
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)

        return {
            'text': review_text,
            'image': image,
            'label': rating
        }


class MultimodalProductIntelligenceModel(nn.Module):
    """Skeleton structure for a Multimodal Model combining Text and Visual features."""
    def __init__(self, num_classes=5, text_embed_dim=256, visual_embed_dim=512):
        super(MultimodalProductIntelligenceModel, self).__init__()
        
        # Example text pipeline layer placeholder (e.g., to process outputs from an LSTM/Transformer layer)
        self.text_projection = nn.Linear(text_embed_dim, 128)
        
        # Example image pipeline projection placeholder (e.g., to process visual features extracted from ResNet/ViT)
        self.image_projection = nn.Linear(visual_embed_dim, 128)
        
        # Combined Classification Head Architecture
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_features, image_features):
        # Extract projection mappings
        t_feat = self.text_projection(text_features)
        i_feat = self.image_projection(image_features)
        
        # Concat embeddings (Multimodal Fusion strategy)
        fused_features = torch.cat((t_feat, i_feat), dim=1)
        
        logits = self.classifier(fused_features)
        return logits