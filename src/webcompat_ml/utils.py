import os

import pandas
import joblib


def extract_gh_labels(obj):
    """Extract labels from GH API responses"""
    return sorted([label["name"] for label in obj["labels"]])


def handle_empty_values(df):
    """Handle empty values in out input dataframe"""
    return df.fillna("EMPTY")


def extract_categorical(df, columns):
    """Extract categorical based on pipeline encoders"""

    for column in columns:
        filename = "{}_LabelEncoder.joblib".format(column)
        path = os.path.join("encoders", filename)
        encoder = joblib.load(path)

        df[column] = encoder.fit_transform(df[column])

    return df


def prepare_gh_event_invalid(obj):
    """Prepare a dataframe to run a prediction task"""

    issue = {
        "body": obj["issue"]["body"],
        "title": obj["issue"]["title"],
        "labels": extract_gh_labels({"labels": obj["issue"]["labels"]}),
        "invalid": "",
    }

    df = pandas.DataFrame([issue])
    df = handle_empty_values(df)
    df = extract_categorical(df, columns=["labels"])

    return df
