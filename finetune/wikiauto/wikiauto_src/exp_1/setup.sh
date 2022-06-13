#!/bin/bash

wget -P ../../data https://www.dropbox.com/sh/ohqaw41v48c7e5p/AAATBDhU1zpdcT5x5WgO8DMaa/wiki-auto-all-data/wiki-auto-part-1-data.json?dl=0

pip install transformers
pip install git+https://github.com/PyTorchLightning/pytorch-lightning
pip install hydra-core --upgrade
pip install wandb
wandb login

python ../../data/prepare_data.py
