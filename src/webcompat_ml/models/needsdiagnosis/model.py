import numpy
import pandas

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class NeedsDiagnosisModel(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self):
        self.xgb_params = {
            "eta": 0.1,
            "max_depth": 7,
            "gamma": 1,
            "min_child_weight": 1,
            "subsample": 0.5,
            "colsample_bytree": 0.8,
            "max_bin": 256,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "silent": 1,
        }
        self.clf = XGBClassifier(**self.xgb_params)
        self.le = LabelEncoder()

    def preprocess(self, X, y):
        corpus = pandas.concat([X["body"], X["title"]], axis=0).tolist()
        self.tokenizer = CountVectorizer(max_features=10000)
        self.tokenizer.fit(corpus)

        needsdiagnosis = self.le.fit_transform(y["needsdiagnosis"])
        body = self.tokenizer.transform(X["body"].values).toarray()
        title = self.tokenizer.transform(X["title"].values).toarray()
        X = numpy.hstack([body, title])
        y = needsdiagnosis
        return (X, y)

    def fit(self, X, y):
        X, y = self.preprocess(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        eval_set = [(X_test, y_test)]
        self.clf.fit(
            X_train,
            y_train,
            early_stopping_rounds=10,
            eval_metric="logloss",
            eval_set=eval_set,
            verbose=True,
        )
        return self

    def predict(self, X):
        body = self.tokenizer.transform(X["body"].values).toarray()
        title = self.tokenizer.transform(X["title"].values).toarray()
        X = numpy.hstack([body, title])
        return self.clf.predict(X)
