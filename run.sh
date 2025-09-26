#!/bin/bash
apt update
apt install python3.10

curl -sSL https://install.python-poetry.org | python3 -

poetry config virtualenvs.in-project true

poetry install

source .venv/bin/activate

wandb login






