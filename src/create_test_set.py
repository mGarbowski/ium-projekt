"""Create test set for A/B experiment."""
import pandas as pd

from listings_preprocessing import aggregate_rating_columns

LISTINGS_FILE = "../data/raw/listings.csv"
TEST_SET_FILE = "../data/processed/test_set.csv"

# TODO make the test set and train set disjoint

def main():
    listings = pd.read_csv(LISTINGS_FILE)
    listings = aggregate_rating_columns(listings)

    columns_absent_in_api_requests = [
        "last_scraped",
        "scrape_id",
        "calendar_updated",
        "calendar_last_scraped",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "first_review",
        "last_review",
        "reviews_per_month",
    ]
    listings = listings.drop(columns=columns_absent_in_api_requests, errors="ignore")
    listings = listings[listings["avg_rating"].notna()]
    listings.to_csv(TEST_SET_FILE, index=False)


if __name__ == "__main__":
    main()
