#!/bin/bash

#requirements install
pip install transformers==3.0.2

#download spanbert pretrained model
cd ./spanbert_hf_base
wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz
tar -xzvf spanbert_hf_base.tar.gz
cd ..

#download a check point from Google Drive
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GrGQzcZyAfd2kHjCZJ0jbIISqMg69s15' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GrGQzcZyAfd2kHjCZJ0jbIISqMg69s15" -O Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt && rm -rf /tmp/cookies.txt