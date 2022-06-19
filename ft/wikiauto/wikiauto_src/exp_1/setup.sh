#!/bin/bash

#wget -P ../../data https://www.dropbox.com/sh/ohqaw41v48c7e5p/AAATBDhU1zpdcT5x5WgO8DMaa/wiki-auto-all-data/wiki-auto-part-1-data.json?dl=0

pip install transformers
pip install git+https://github.com/PyTorchLightning/pytorch-lightning
pip install hydra-core --upgrade
pip install wandb
wandb login

#python ../../data/prepare_data.py

#download wikiauto_dataframe.pickle 
wget -P ../../data --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AEFplcg5_lE5plDKzoLzXEL3GqB-q23d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AEFplcg5_lE5plDKzoLzXEL3GqB-q23d" -O wikiauto_dataframe.pickle && rm -rf /tmp/cookies.txt
