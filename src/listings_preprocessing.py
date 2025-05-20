"""Utilities for transforming listings data."""

from __future__ import annotations

import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.imputation import impute_missing_values
from src.schema import Listing


def drop_useless_columns(df, debug=True):
    cols_to_drop = [
        "id",
        "listing_url",
        "scrape_id",
        "last_scraped",
        "source",
        "picture_url",
        "host_name",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "calendar_last_scraped",
        "calendar_updated",  # only missing values
        "neighbourhood",
        "host_location",
        "host_neighbourhood",
        "neighbourhood_cleansed",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "first_review",  # will not be present in new offers
        "last_review",  # will not be present in new offers
        "latitude",  # neighbourhood category should be more informative
        "longitude",  # neighbourhood category should be more informative
        "amenities",
        # this could be useful but has too many categories (1000+), maybe use one-hot encoding + embeddings?
        "reviews_per_month",  # will be missing for new offers
    ]

    if not debug:
        return df.drop(columns=cols_to_drop, errors="ignore")
    return df.drop(columns=cols_to_drop)


def drop_fulltext_columns(df, debug=True):
    """Text columns that are useless but could be processed with an LLM(?)"""
    text_columns = [
        "name",
        "description",
        "neighborhood_overview",
        "host_about",
        "license",
    ]

    if not debug:
        return df.drop(columns=text_columns, errors="ignore")

    return df.drop(columns=text_columns)


def transform_binary_columns(df, debug=True):
    """To standard 0/1"""
    binary_columns = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "has_availability",
        "instant_bookable",
    ]

    for c in binary_columns:
        if debug:
            unique_vals = df[c].unique()
            assert len(unique_vals) == 2 or len(unique_vals) == 3
            assert "t" in unique_vals and "f" in unique_vals

        df[c] = df[c].apply(lambda x: 1 if x == "t" else 0)

    return df


def aggregate_rating_columns(df):
    """Replace all review scores columns with a single average rating column"""
    rating_columns = [
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]
    df["avg_rating"] = df[rating_columns].mean(axis=1)
    df = df.drop(columns=rating_columns)
    return df


def add_average_rating_by_host(df):
    """Add average rating by host"""
    # FIXME this includes the record itself - predicted variable leaks to the training set
    df["avg_rating_by_host"] = df.groupby("host_id")["avg_rating"].transform("mean")
    return df


def transform_price(df):
    """Transform price column to float"""
    transform_string_price = lambda p: float(p.replace("$", "").replace(",", ""))

    df["price"] = df["price"].apply(
        lambda p: transform_string_price(p) if isinstance(p, str) else p
    )
    return df


def get_unique_values_in_list_column(df, column_name):
    all_values = set()
    for row in df[column_name]:
        if not isinstance(row, str):
            continue
        row_values = attribute_value_to_list(row)
        row_values = [a.strip() for a in row_values]
        all_values.update(row_values)

    return all_values


def attribute_value_to_list(value):
    if not isinstance(value, str):
        return []
    values = (
        value.replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .replace("'", "")
        .split(",")
    )
    values = [a.strip() for a in values]
    return values


def one_hot_encode_list_column(df, column_name):
    """One-hot encode list column"""
    all_values = get_unique_values_in_list_column(df, column_name)
    for value in all_values:
        df[f"{column_name}_{value}"] = df[column_name].apply(
            lambda x: True if value in attribute_value_to_list(x) else False
        )
    df.drop(columns=[column_name], inplace=True)
    return df


def transform_host_verifications(df):
    """Explicit instead of one_hot_encode_list_column"""
    col_name = "host_verifications"
    expected_values = ["", "email", "phone", "work_email"]
    row_values = attribute_value_to_list(df[col_name])
    for row_value in row_values:
        assert (
            row_value in expected_values
        ), f"Unexpected value {row_value} in column {col_name}"

    for expected_value in expected_values:
        df[f"{col_name}_{expected_value}"] = expected_value in row_values

    df = df.drop(columns=[col_name])
    return df


def extract_host_country(df):
    """Extract host country from host_location"""

    def extract_country_from_location(location: str):
        if not "," in location:
            return location

        state = location.split(",")[-1].strip()

        if len(state) == 2:
            return "United States"

        return state

    df["host_country"] = df["host_location"].apply(
        lambda x: extract_country_from_location(x) if isinstance(x, str) else x
    )
    return df


def transform_host_response_time(df):
    """Transform host response time to a number"""
    # TODO adjust scale
    scale = {
        "within an hour": 1,
        "within a few hours": 2,
        "within a day": 3,
        "a few days or more": 4,
    }

    df["host_response_time"] = df["host_response_time"].apply(
        lambda x: scale.get(x, np.nan)
    )
    return df


def group_property_types(df):
    """Group together similar property types"""

    def transform(property_type: str):
        if "entire rental unit" in property_type.lower():
            return "entire rental unit"
        elif "room" in property_type.lower():
            return "room"
        elif "apartment" in property_type.lower():
            return "apartment"
        elif "home" in property_type.lower():
            return "home"
        elif "condo" in property_type.lower():
            return "condo"
        else:
            return "other"

    df["property_type"] = df["property_type"].apply(
        lambda x: transform(x) if isinstance(x, str) else x
    )
    return df


def extract_num_bathrooms(df):
    def parse_bathroom(text):
        text = str(text).lower()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        elif "half" in text:
            return 0.5
        else:
            return None

    df["num_bathrooms"] = df["bathrooms_text"].apply(parse_bathroom)
    return df


def extract_is_shared_from_bathrooms_text(df):
    """Extract is_shared from bathrooms_text column"""
    extract_is_shared = lambda txt: 1 if "shared" in txt.lower() else 0
    df["is_shared_bathroom"] = df["bathrooms_text"].apply(
        lambda x: extract_is_shared(x) if isinstance(x, str) else x
    )
    df = df.drop(columns=["bathrooms_text"])
    return df


def categorical_columns_one_hot_encoding(df, debug=True):
    """One-hot encode categorical columns"""
    categorical_columns = {
        "neighbourhood_group_cleansed": [
            "Neukölln",
            "Pankow",
            "Tempelhof - Schöneberg",
            "Mitte",
            "Friedrichshain-Kreuzberg",
            "Treptow - Köpenick",
            "Lichtenberg",
            "Reinickendorf",
            "Charlottenburg-Wilm.",
            "Steglitz - Zehlendorf",
            "Marzahn - Hellersdorf",
            "Spandau",
        ],
        "property_type": [
            "entire rental unit",
            "room",
            "condo",
            "other",
            "apartment",
            "home",
        ],
        "room_type": [
            "Entire home/apt",
            "Private room",
            "Hotel room",
            "Shared room",
        ],
        "is_shared_bathroom": [1.0, 0.0, np.nan],
    }

    if debug:
        for column, values in categorical_columns.items():
            unique_vals = df[column].unique()
            # Filter out NaN values for comparison
            unique_vals_without_nan = [val for val in unique_vals if not pd.isna(val)]
            expected_values_without_nan = [val for val in values if not pd.isna(val)]

            assert len(unique_vals_without_nan) == len(
                expected_values_without_nan
            ), f"Column {column} has {len(unique_vals_without_nan)} unique values, expected {len(expected_values_without_nan)}"
            assert set(unique_vals_without_nan) == set(
                expected_values_without_nan
            ), f"Actual: {unique_vals_without_nan}, expected: {expected_values_without_nan}"

            # Check if NaN exists in both actual and expected values
            assert (
                pd.isna(unique_vals).any() == pd.isna(values).any()
            ), f"NaN presence mismatch in column {column}"

            if pd.isna(values).any():
                df[column] = df[column].fillna(
                    "nan"
                )  # ensure a separate column gets created for NaN values

        df = pd.get_dummies(df, columns=list(categorical_columns.keys()))
        return df
    else:
        for column, values in categorical_columns.items():
            target_columns = [f"{column}_{value}" for value in values]
            encoded_column = pd.get_dummies(df[column], columns=[column])
            encoded_column = encoded_column.reindex(
                columns=target_columns, fill_value=0
            )
            df = pd.concat([df, encoded_column], axis=1)
            df = df.drop(columns=[column])
        return df


def transform_host_since(df):
    """Transform timestamp to number of years since"""
    current_year = datetime.now().year

    extract_years_since = lambda timestamp: current_year - int(timestamp.split("-")[0])

    df["host_since"] = df["host_since"].apply(
        lambda x: extract_years_since(x) if isinstance(x, str) else x
    )
    return df


def transform_percentage_to_number(df):
    """Transform percentage to number from range [0, 1]"""
    percentage_columns = [
        "host_response_rate",
        "host_acceptance_rate",
    ]
    transform_percentage = lambda x: float(x.replace("%", "")) / 100

    for c in percentage_columns:
        df[c] = df[c].apply(
            lambda x: transform_percentage(x) if isinstance(x, str) else x
        )
    return df


def normalize_numerical_columns(df, scaler_file, load=False):
    """Standardize numerical columns to have mean 0 and std 1"""

    numerical_columns = [
        "maximum_nights",
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "price",
    ]

    if not load:  # Create a new one and save to file
        scaler = StandardScaler()
        scaler.fit(df[numerical_columns])

        with open(scaler_file, "wb") as file:
            pickle.dump(scaler, file)
    else:
        # Load the scaler from file
        with open(scaler_file, "rb") as file:
            scaler: StandardScaler = pickle.load(file)

    df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df


def drop_rows_with_no_rating(df):
    assert "avg_rating" in df.columns
    df = df.dropna(subset=["avg_rating"])
    return df


def drop_cols_absent_in_api_requests(df):
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
    df = df.drop(columns=columns_absent_in_api_requests, errors="ignore")
    return df


def transform_listings(df, scaler_file, imputer_file, impute=False):
    """For transforming dataframe containing the entire dataset"""
    df = drop_useless_columns(df)
    df = drop_fulltext_columns(df)
    df = transform_binary_columns(df)
    df = transform_price(df)
    df = transform_host_response_time(df)
    df = extract_num_bathrooms(df)
    df = extract_is_shared_from_bathrooms_text(df)
    df = group_property_types(df)
    df = categorical_columns_one_hot_encoding(
        df
    )  # after extracting is_shared_bathroom and grouping property types!
    df = transform_host_since(df)
    df = transform_percentage_to_number(df)
    df = one_hot_encode_list_column(df, "host_verifications")
    # df = add_average_rating_by_host(df)
    df = df.drop(columns=["host_id"])  # after adding avg_rating_by_host
    df = normalize_numerical_columns(df, scaler_file, load=False)
    df = df.sort_index(axis=1)  # sort columns to have a consistent order

    if impute:
        df = impute_missing_values(df, imputer_file, load=False)
    return df


def transform_item(df, scaler_file, imputer_file, impute=False):
    """For transforming dataframe with a single item"""
    df = drop_useless_columns(df, debug=False)
    df = drop_fulltext_columns(df, debug=False)
    df = transform_binary_columns(df, debug=False)
    df = transform_price(df)
    df = transform_host_response_time(df)
    df = extract_num_bathrooms(df)
    df = extract_is_shared_from_bathrooms_text(df)
    df = group_property_types(df)
    df = categorical_columns_one_hot_encoding(df, debug=False)
    df = transform_host_since(df)
    df = transform_percentage_to_number(df)
    df = transform_host_verifications(df)
    df = df.drop(columns=["host_id"])
    df = df.sort_index(axis=1)
    df = normalize_numerical_columns(df, scaler_file, load=True)
    if impute:
        df = impute_missing_values(df, imputer_file, load=True)
    return df


class ListingTransformer:
    def __init__(self, scaler_file: str, imputer_file: str):
        self.scaler_file = scaler_file
        self.imputer_file = imputer_file

    def transform(self, listing: Listing) -> pd.DataFrame:
        df = self.convert_to_dataframe(listing)
        return transform_item(df, self.scaler_file, self.imputer_file, impute=True)

    def convert_to_dataframe(self, listing: Listing) -> pd.DataFrame:
        """Convert a Listing object to a DataFrame."""
        return pd.DataFrame([listing.model_dump()])
