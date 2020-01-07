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

## MyPy

This package uses MyPy for Python type checking and intellisense.

To install mypy in vscode, install the 'mypy' plugin and run these:

```sh
sudo apt install python3.8-venv python3.8-dev
python3.8 -m venv ~/.mypyls
~/.mypyls/bin/pip install -U wheel
~/.mypyls/bin/pip install -U "https://github.com/matangover/mypyls/archive/master.zip#egg=mypyls[patched-mypy]"
```
