# topic-classification

Learn and apply paragraph to topic training.

## Installation

This code requires Python 3.7 venv and pip.

To install, run `./run install`.

**Optional**: Install GPU support with `./run pip install tensorflow-gpu==1.15.0`.

## Setup

The code requires a BERT(-like) model to produce sequence (sentence / paragraph) embeddings.

A good starting point is "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1".

Model names starting with "http" are retrieved from tfhub, others are loaded from the given local path.

## Running

To run the web server, `./run server`.

During development, use `./run devserver`.

## Testing

To run operations from the command line, `./run local --help`.

To run nose tests, `./run test`.

To run pycodestyle, `./run lint`.
