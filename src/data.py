import os
import requests
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


def initialize_dirs():
    """Create data directories if they don't exist."""
    Path('data').mkdir(exist_ok=True)
    Path('data/images').mkdir(exist_ok=True)
    Path('data/splits').mkdir(exist_ok=True)
    print('✅ Setup complete.')


def load_raw_data():
    """Load reviews and metadata from HuggingFace Hub."""
    print('Loading reviews...')
    reviews_ds = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        'raw_review_All_Beauty',
        split='full',
        trust_remote_code=True
    )
    
    print('Loading metadata...')
    meta_ds = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        'raw_meta_All_Beauty',
        split='full',
        trust_remote_code=True
    )
    
    return reviews_ds.to_pandas(), meta_ds.to_pandas()


def extract_image_url(images_dict):
    """Helper to safely parse nested image dictionary strings."""
    try:
        for key in ['hi_res', 'large']:
            for url in (images_dict.get(key) or []):
                if url and str(url) != 'None':
                    return url
    except:
        pass
    return None


def clean_data(reviews_df, meta_df):
    """Clean metadata and review DataFrames."""
    # 1. Clean Metadata
    meta_clean = meta_df[[
        'parent_asin', 'title', 'average_rating', 'rating_number',
        'price', 'store', 'categories', 'features', 'description', 'images'
    ]].copy()

    meta_clean['price'] = (
        meta_clean['price'].astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', np.nan).astype(float)
    )
    meta_clean = meta_clean.drop_duplicates(subset='parent_asin', keep='first')
    meta_clean['image_url'] = meta_clean['images'].apply(extract_image_url)

    # 2. Clean Reviews
    reviews_clean = reviews_df[[
        'parent_asin', 'rating', 'title', 'text', 'helpful_vote', 'verified_purchase', 'timestamp'
    ]].copy()
    reviews_clean = reviews_clean.dropna(subset=['text', 'rating'])
    reviews_clean['review_length'] = reviews_clean['text'].str.split().str.len()

    valid_asins = set(meta_clean['parent_asin'])
    reviews_clean = reviews_clean[reviews_clean['parent_asin'].isin(valid_asins)]
    
    return reviews_clean, meta_clean


def subsample_and_split(reviews_clean, meta_clean, max_products=15000, min_reviews=3):
    """Filter by review counts, cap dataframe size, and split by product."""
    counts = reviews_clean['parent_asin'].value_counts()
    active = counts[counts >= min_reviews].index
    reviews_clean = reviews_clean[reviews_clean['parent_asin'].isin(active)]
    meta_clean = meta_clean[meta_clean['parent_asin'].isin(active)]

    if meta_clean['parent_asin'].nunique() > max_products:
        keep = meta_clean['parent_asin'].sample(max_products, random_state=SEED).values
        meta_clean = meta_clean[meta_clean['parent_asin'].isin(keep)]
        reviews_clean = reviews_clean[reviews_clean['parent_asin'].isin(keep)]

    # Split by Product to prevent target leakage
    all_asins = meta_clean['parent_asin'].unique()
    train_asins, temp = train_test_split(all_asins, test_size=0.30, random_state=SEED)
    val_asins, test_asins = train_test_split(temp, test_size=0.50, random_state=SEED)

    # Save components
    pd.Series(train_asins).to_csv('data/splits/train_asins.csv', index=False)
    pd.Series(val_asins).to_csv('data/splits/val_asins.csv', index=False)
    pd.Series(test_asins).to_csv('data/splits/test_asins.csv', index=False)
    
    meta_clean.to_parquet('data/meta_clean.parquet', index=False)
    reviews_clean.to_parquet('data/reviews_clean.parquet', index=False)
    
    print(f'✅ Train: {len(train_asins)} | Val: {len(val_asins)} | Test: {len(test_asins)}')
    return meta_clean


def download_product_images(meta_clean, max_images=8000):
    """Download image URLs, resize them to 224x224, and save locally."""
    pool = meta_clean[meta_clean['image_url'].notna()]
    pool = pool.sample(min(max_images, len(pool)), random_state=SEED)

    success, failed = 0, 0
    for _, row in tqdm(pool.iterrows(), total=len(pool), desc='Downloading Images'):
        path = Path(f'data/images/{row["parent_asin"]}.jpg')
        if path.exists():
            success += 1
            continue
        try:
            r = requests.get(row['image_url'], timeout=8)
            img = Image.open(BytesIO(r.content)).convert('RGB').resize((224, 224))
            img.save(path, 'JPEG', quality=85)
            success += 1
        except:
            failed += 1

    print(f'✅ Downloaded: {success:,} | Failed: {failed:,}')


def run_pipeline():
    """Orchestrates the entire processing script."""
    initialize_dirs()
    reviews_df, meta_df = load_raw_data()
    reviews_clean, meta_clean = clean_data(reviews_df, meta_df)
    meta_final = subsample_and_split(reviews_clean, meta_clean)
    download_product_images(meta_final)


if __name__ == '__main__':
    run_pipeline()