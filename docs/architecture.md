# Architecture

## Code structure

* All code goes in the `/src/` directory
* Bundled package data
  * Binary models (pre-trained)
    * `src/models/<model>/model.bin`
  * Encoders for processing input
    * `src/models/<model>/encoders/`
  * Metadata about model metrics
    * `src/models/<model>/metadata/results.csv`
* Models
  * Each model is a python module under `webcompat_ml.models`
  * Each model has a CLI tool for training and classification
  * Each CLI is installed as a console script so the rest of the pipeline can use it as a standalone program.
* Utils
  * We keep track of some helpers functions used by the rest of the pipeline in the `webcompat_ml.utils` module.

## Available models

### Invalid

* Classifies issues as `valid` or `invalid`.
* Based on `xgboost`

## Pipeline

### Data processing

* Given a pandas `DataFrame`
  * Build and store the encoders for fields when loading data required for the model
  * Load the encoders defined in the building process

### Training

* Given a `CSV` as input
  * Convert input to a pandas `DataFrame`
  * Load the encoders defined in the building phase
  * Process data using the encoders
  * Split the input to training/validation sets
  * Train the model with training data set
  * Validate the model against validation data set
  * Store model metrics

### Classification

* Given a `CSV` as input and a trained model
  * Convert input to a pandas `DataFrame`
  * Load the encoders defined in the building phase
  * Process data using the encoders
  * Generate predictions