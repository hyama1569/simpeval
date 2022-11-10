#!/bin/bash

#requirements install
pip install wandb
wandb login
pip install git+https://github.com/facebookresearch/text-simplification-evaluation.git
pip install transformers
pip install pytorch-lightning
pip install hydra-core --upgrade