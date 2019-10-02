import numpy
import pandas

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class NeedsDiagnosisModel(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Model to predict needsdiagnosis flags"""

    def __init__(self, verbose=True):
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
        self.verbose = verbose

    def preprocess(self, X, y):
        """Preprocess data

        * body, title: Tokenize input
        * needsdiagnosis: Encode labels

        """

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
        """Fit the XGBClassifier used for the model"""

        X, y = self.preprocess(X, y)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3)
        eval_set = [(X_eval, y_eval)]
        self.clf.fit(
            X_train,
            y_train,
            early_stopping_rounds=10,
            eval_metric="logloss",
            eval_set=eval_set,
            verbose=self.verbose,
        )

        y_pred = self.clf.predict(X_eval)

        if self.verbose:
            print(classification_report(y_eval, y_pred))
            print(confusion_matrix(y_eval, y_pred))

        return self

    def predict(self, X):
        """Predict needsdiagnosis flags"""
        body = self.tokenizer.transform(X["body"].values).toarray()
        title = self.tokenizer.transform(X["title"].values).toarray()
        X = numpy.hstack([body, title])
        return self.clf.predict(X)

    def predict_proba(self, X):
        body = self.tokenizer.transform(X["body"].values).toarray()
        title = self.tokenizer.transform(X["title"].values).toarray()
        X = numpy.hstack([body, title])
        return self.clf.predict_proba(X)
