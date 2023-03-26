#!/bin/bash

#requirements install
pip install wandb
wandb login
#pip install git+https://github.com/facebookresearch/text-simplification-evaluation.git
pip install transformers
pip install pytorch-lightning==1.8.1
pip install pandas==1.4.1
pip install hydra-core --upgrade