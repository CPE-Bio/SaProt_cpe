import argparse
import pandas as pd
import lmdb
import json
import os
import sklearn
from sklearn.model_selection import train_test_split

def create_dataset(path, data):
    """
    Creates an LMDB dataset from a list of dictionaries.
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Open LMDB environment
    # map_size set to 10GB to accommodate larger datasets
    env = lmdb.open(path, map_size=10 * 1024 * 1024 * 1024)
    
    with env.begin(write=True) as txn:
        for i, entry in enumerate(data):
            # Key: Index as bytes (e.g., b'0', b'1')
            key = str(i).encode('utf-8')
            # Value: JSON string as bytes
            value = json.dumps(entry).encode('utf-8')
            txn.put(key, value)
        
        # Special Key: 'length' stores the dataset size
        txn.put(b'length', str(len(data)).encode('utf-8'))
    
    env.close()
    print(f"Created LMDB dataset at {path} with {len(data)} samples.")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to SaProt LMDB format")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LMDBs")
    parser.add_argument("--column", type=str, default="Sequence", help="Column name for sequences")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset to include in the test split")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Proportion of dataset to include in the validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    print(f"Reading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in CSV. Available columns: {list(df.columns)}")

    # Extract sequences and format as list of dicts
    # Ensure sequences are strings and handle potential NaNs
    sequences = df[args.column].dropna().astype(str).tolist()
    data = [{"seq": seq} for seq in sequences]
    
    print(f"Total samples: {len(data)}")

    # Split data
    # First split off test set
    train_val_data, test_data = train_test_split(data, test_size=args.test_size, random_state=args.seed)
    
    # Adjust valid size relative to the remaining data
    remaining_prop = 1.0 - args.test_size
    valid_prop = args.valid_size / remaining_prop
    
    train_data, valid_data = train_test_split(train_val_data, test_size=valid_prop, random_state=args.seed)

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create LMDBs
    create_dataset(os.path.join(args.output_dir, "train"), train_data)
    create_dataset(os.path.join(args.output_dir, "valid"), valid_data)
    create_dataset(os.path.join(args.output_dir, "test"), test_data)

if __name__ == "__main__":
    main()
