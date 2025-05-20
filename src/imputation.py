import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def impute_missing_values(df: pd.DataFrame, imputer_file: str, load: bool = False) -> pd.DataFrame:
    """Impute missing values in the dataframe using saved or new imputers"""

    def create_preprocessor(df_no_target: pd.DataFrame):
        # Separate binary and numerical cols
        binary_cols = [
            col
            for col in df_no_target.columns
            if df_no_target[col]
            .dropna()
            .isin([0, 1])
            .all()  # pyright: ignore[reportGeneralTypeIssues]
            and df_no_target[col].nunique(dropna=True) <= 2
        ]
        numerical_cols = (
            df_no_target.select_dtypes(include=["number"])
            .columns.difference(binary_cols)
            .tolist()
        )

        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        return (
            ColumnTransformer(
                transformers=[
                    ("num", num_imputer, numerical_cols),
                    ("cat", cat_imputer, binary_cols),
                ]
            ),
            numerical_cols + binary_cols,
        )

    df_no_target = df.drop(columns=["avg_rating"], errors="ignore")

    if load:
        with open(imputer_file, "rb") as file:
            preprocessor = pickle.load(file)
        columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
    else:
        preprocessor, columns = create_preprocessor(df_no_target)
        preprocessor.fit(df_no_target)
        with open(imputer_file, "wb") as file:
            pickle.dump(preprocessor, file)

    # Impute and reconstruct DataFrame
    imputed_array = preprocessor.transform(df_no_target)
    imputed_df = pd.DataFrame(
        imputed_array,
        columns=columns,  # pyright: ignore[reportArgumentType]
        index=df.index,
    )

    assert (
        not imputed_df.isna().any().any()  # pyright: ignore[reportAttributeAccessIssue]
    ), "Data still contains NaN values after imputation"
    assert imputed_df.shape[0] == df.shape[0], "Row count mismatch after imputation"

    # Add the target column back
    if "avg_rating" in df.columns:
        imputed_df["avg_rating"] = df["avg_rating"]

    return imputed_df
