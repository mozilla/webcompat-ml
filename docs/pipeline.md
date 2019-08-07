# Pipeline
## Introduction

The purpose of the [pipeline](../pipeline.py) is, given a Pandas DataFrame as input (`df`) with data from
WebCompat issues, to allow:

* Building the encoders of the model fields
* Storing the encoders in `json` format for future usage
* Loading the encoders for training or running predictions
* Train the model
* Store trained model in binary format
* Use a trained model to run predictions
