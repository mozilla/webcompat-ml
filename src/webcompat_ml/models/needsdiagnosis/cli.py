import click
import joblib
import pandas

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from webcompat_ml.models.needsdiagnosis import NeedsDiagnosisModel


@click.group()
def main():
    pass


@main.command()
@click.option("--data", help="Path to dataset CSV")
@click.option("--output", help="Path to model binary path", default="model.bin")
def train(data, output):
    dataset = pandas.read_csv(data)
    X = dataset[["body", "title"]]
    y = dataset[["needsdiagnosis"]]
    model = NeedsDiagnosisModel()
    model.fit(X, y)
    joblib.dump(model, output)


@main.command()
@click.option("--data", help="Path to input CSV")
@click.option("--model", help="Path to binary model")
@click.option("--output", help="Predictions output")
def predict(data, model, output):
    X = pandas.read_csv(data)
    model = joblib.load(model)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    y_pred_labels = model.le.inverse_transform(y_pred)

    output_predictions = pandas.DataFrame(
        data=y_pred_labels, columns=["needsdiagnosis"]
    )
    output_probas = pandas.DataFrame(data=y_pred_proba, columns=model.le.classes_)
    output_probas = output_probas.add_prefix("proba_")
    output_df = pandas.concat([output_predictions, output_probas], axis=1)
    output_df.to_csv(output, index=False)


@main.command()
@click.option("--data", help="Path to input CSV")
def evaluate(data):
    dataset = pandas.read_csv(data)
    kf = KFold(n_splits=3)
    for train_index, test_index in kf.split(dataset):
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        model = NeedsDiagnosisModel(verbose=False)
        X_train = train[["body", "title"]]
        y_train = train[["needsdiagnosis"]]
        X_test = test[["body", "title"]]
        y_test = test[["needsdiagnosis"]]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
