"""Script for splitting and transforming the dataset.

We will be using only those records with non-null avg_rating.

listings.csv will be split into train and test set

Test set will be saved in 2 variants
- for model development - same processing as training set
- for A/B testing - minimal processing, used for generating API requests
"""

import numpy as np
import pandas as pd

from src.listings_preprocessing import (
    transform_listings,
    aggregate_rating_columns,
    drop_rows_with_no_rating,
    drop_cols_absent_in_api_requests,
)

LISTINGS_FILE = "data/raw/listings.csv"
TRAIN_SET_FILE = "data/processed/train.csv"
TEST_SET_FILE = "data/processed/test_set.csv"
EXPERIMENT_SET_FILE = "data/processed/experiment_set.csv"

SCALER_FILE = "models/scaler.pkl"
IMPUTER_FILE = "models/imputer_pipeline.pkl"


def main():
    test_ratio = 0.2
    listings = pd.read_csv(LISTINGS_FILE)

    # Calculate target column and drop rows where it is empty
    listings = aggregate_rating_columns(listings)
    listings = drop_rows_with_no_rating(listings)

    # Split into train and test set
    total_size = len(listings)
    np.random.seed(42)
    test_indices = np.random.choice(
        total_size, size=int(total_size * test_ratio), replace=False
    )
    train_indices = np.setdiff1d(np.arange(total_size), test_indices)

    for i in range(total_size):
        if i in test_indices:
            assert i not in train_indices
        else:
            assert i in train_indices

    print(f"Total size: {total_size}")
    print(f"Train set size: {len(train_indices)}")
    print(f"Test set size: {len(test_indices)}")

    # Save experiment set for A/B testing
    experiment_set = listings.iloc[test_indices]
    experiment_set = drop_cols_absent_in_api_requests(experiment_set)
    experiment_set.to_csv(EXPERIMENT_SET_FILE, index=False)
    print(f"Experiment set size: {len(experiment_set)}")
    print(f"Experiment set saved to {EXPERIMENT_SET_FILE}")

    # Process and save train and test sets
    listings = transform_listings(listings, SCALER_FILE, IMPUTER_FILE, impute=False)

    train_set = listings.iloc[train_indices]
    train_set.to_csv(TRAIN_SET_FILE, index=False)
    print(f"Train set saved to {TRAIN_SET_FILE}")

    test_set = listings.iloc[test_indices]
    test_set.to_csv(TEST_SET_FILE, index=False)
    print(f"Test set saved to {TEST_SET_FILE}")


if __name__ == "__main__":
    main()
