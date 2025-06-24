"""
Preprocess Douban Movies Dataset to RecBole format
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
from datetime import datetime

def convert_timestamp(time_str):
    """Convert time string to Unix timestamp"""
    try:
        if pd.isna(time_str) or time_str == '' or time_str.strip() == '':
            return 0
        # Parse the date string (format: YYYY-MM-DD)
        dt = datetime.strptime(str(time_str).strip(), '%Y-%m-%d')
        return int(dt.timestamp())
    except:
        return 0

def preprocess_movies_only():
    """Preprocess movies data to RecBole format"""
    BASE_PATH = "" # Set this to the base path of your dataset 
    
    # Define paths
    data_dir = os.path.join(BASE_PATH, "douban_dataset(text information)")
    output_dir = os.path.join(BASE_PATH, "movies_recbole")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== PROCESSING MOVIES DATA ===")
    
    # Load movie metadata
    print("Loading movie metadata...")
    movies_df = pd.read_csv(os.path.join(data_dir, "movies_cleaned.txt"), sep='\t', quoting=1)
    print(f"Loaded {len(movies_df)} movies")
    print(f"Movie columns: {list(movies_df.columns)}")
    
    # Load movie reviews (interactions)
    print("\nLoading movie reviews...")
    movie_reviews_df = pd.read_csv(os.path.join(data_dir, "moviereviews_cleaned.txt"), sep='\t', quoting=1)
    print(f"Loaded {len(movie_reviews_df)} movie reviews")
    print(f"Review columns: {list(movie_reviews_df.columns)}")
    
    # Clean and prepare interactions data
    print("\nProcessing interactions...")
    interactions = []
    
    for _, row in tqdm(movie_reviews_df.iterrows(), total=len(movie_reviews_df), desc="Processing movie reviews"):
        try:
            user_id = str(row['user_id']).strip('"')
            item_id = str(row['movie_id']).strip('"')
            rating = float(row['rating'])
            timestamp = convert_timestamp(row.get('time', ''))
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
        except Exception as e:
            continue
    
    interactions_df = pd.DataFrame(interactions)
    print(f"Processed {len(interactions_df)} valid interactions")
    
    # Get statistics without remapping
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_items = sorted(interactions_df['item_id'].unique())
    
    print(f"Unique users: {len(unique_users)}")
    print(f"Unique items: {len(unique_items)}")
    
    # Create .inter file with original IDs
    print("\nCreating .inter file...")
    inter_data = []
    for _, row in interactions_df.iterrows():
        inter_data.append({
            'user_id:token': row['user_id'],
            'item_id:token': row['item_id'],
            'rating:float': row['rating'],
            'timestamp:float': row['timestamp']
        })
    
    inter_df = pd.DataFrame(inter_data)
    inter_output_path = os.path.join(output_dir, "movies.inter")
    inter_df.to_csv(inter_output_path, sep='\t', index=False)
    print(f"Saved .inter file: {inter_output_path}")
    
    # Create .item file
    print("\nCreating .item file...")
    item_data = []
    
    for _, movie in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Processing movie metadata"):
        try:
            # Use UID instead of movie_id
            uid = int(str(movie['UID']).strip('"'))
            
            # Clean and extract movie information
            name = str(movie.get('name', '')).strip('"')
            tags = str(movie.get('tag', '')).strip('"')
            category_id = str(movie.get('CategoryID', '')).strip('"')
            
            item_data.append({
                'movie_id:token': uid,
                'name:token': name,
                'tag:token_seq': tags,
                'categoryID:token': category_id
            })
        except Exception as e:
            continue
    
    items_df = pd.DataFrame(item_data)
    item_output_path = os.path.join(output_dir, "movies.item")
    items_df.to_csv(item_output_path, sep='\t', index=False)
    print(f"Saved .item file: {item_output_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total users: {len(unique_users)}")
    print(f"Total movies: {len(unique_items)}")
    print(f"Total interactions: {len(interactions_df)}")
    print(f"Movies with metadata: {len(items_df)}")
    print(f"Average rating: {interactions_df['rating'].mean():.2f}")
    print(f"Rating range: {interactions_df['rating'].min():.1f} - {interactions_df['rating'].max():.1f}")
    print(f"Date range: {interactions_df[interactions_df['timestamp'] > 0]['timestamp'].min():.0f} - {interactions_df[interactions_df['timestamp'] > 0]['timestamp'].max():.0f}")
    
    print(f"\nFiles saved in: {output_dir}")
    print("- movies.inter (interactions)")
    print("- movies.item (movie metadata)")

if __name__ == "__main__":
    preprocess_movies_only()
