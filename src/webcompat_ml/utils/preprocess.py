import os

from importlib.resources import path as importlib_path

import pandas
import joblib


def extract_gh_labels(obj):
    """Extract labels from GH API responses"""
    labels = sorted([label["name"] for label in obj["labels"]])
    return " ".join(labels)


def handle_empty_values(df):
    """Handle empty values in out input dataframe"""
    return df.fillna("EMPTY")


def extract_categorical(df, columns):
    """Extract categorical based on pipeline encoders"""

    for column in columns:
        filename = "{}_LabelEncoder.joblib".format(column)
        module = 'webcompat_ml.utils.encoders'

        with importlib_path(module, filename) as p:
            encoder = joblib.load(p)
            df[column] = encoder.fit_transform(df[column])

    return df


def prepare_gh_event_invalid(obj):
    """Prepare a dataframe to run a prediction task"""

    issue = {
        "body": obj["body"],
        "title": obj["title"],
        "labels": extract_gh_labels({"labels": obj["labels"]}),
        "invalid": "",
    }

    df = pandas.DataFrame([issue])
    df = handle_empty_values(df)
    df = extract_categorical(df, columns=["labels"])

    return df
