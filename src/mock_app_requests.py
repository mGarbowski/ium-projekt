import numpy as np
import pandas as pd
import requests

TEST_SET_FILE = "../data/processed/test_set.csv"
API_ENDPOINT = "http://localhost:8000/predict/"

# The hashes evaluate to their respective model indices
RANDOM_MODEL_UID = "617d32dc-5c87-4ac6-a89d-b991624c529b"
REAL_MODEL_UID = "617d32dc-5c87-4ac6-a89d-b991624c529e"


def row_to_dict(row):
    d = row.to_dict()
    for key, value in row.items():
        if isinstance(value, float) and np.isnan(value):
            d[key] = None

    number_to_string_columns = [
        "id",
        "host_id",
    ]
    for column in number_to_string_columns:
        if column in d:
            d[column] = str(d[column])

    return d

def make_request(listing, user_id):
    payload = {
        "user_id": user_id,
        "data": listing,
    }
    response = requests.post(API_ENDPOINT, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code} {response.json()}")

def main():
    df = pd.read_csv(TEST_SET_FILE)
    df = df.drop(columns=["avg_rating"], errors="ignore")

    # Random model requests
    for _, row in df.iterrows():
        listing = row_to_dict(row)
        make_request(listing, RANDOM_MODEL_UID)


    # Real model requests
    for _, row in df.iterrows():
        listing = row_to_dict(row)
        make_request(listing, REAL_MODEL_UID)


if __name__ == "__main__":
    main()