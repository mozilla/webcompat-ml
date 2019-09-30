import click
import joblib
import pandas

from webcompat_ml.models.needsdiagnosis import NeedsDiagnosisModel


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data", help="Path to dataset CSV")
@click.option("--output", help="Path to model binary path", default="model.bin")
def train(data, output):
    dataset = pandas.read_csv(data)
    X = dataset[["body", "title"]]
    y = dataset[["needsdiagnosis"]]
    model = NeedsDiagnosisModel()
    model.fit(X, y)
    joblib.dump(model, output)


@cli.command()
@click.option("--data", help="Path to input CSV")
@click.option("--model", help="Path to binary model")
@click.option("--output", help="Predictions output")
def predict(data, model, output):
    X = pandas.read_csv(data)
    model = joblib.load(model)
    predictions = model.predict(X)
    predictions = pandas.DataFrame(data=predictions, columns=["predictions"])
    predictions.to_csv("output", index=False)


if __name__ == "__main__":
    cli()
