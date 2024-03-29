#!/bin/bash

#requirements install
pip install wandb
wandb login
pip install git+https://github.com/facebookresearch/text-simplification-evaluation.git
pip install transformers
#pip install -U pip setuptools wheel
pip install setuptools==59.5.0
pip install -U spacy
pip install git+https://github.com/PyTorchLightning/pytorch-lightning
pip install hydra-core --upgrade
python -m spacy download en_core_web_sm
pip install torch==1.9+cu111 -f https://download.pytorch.org/whl/torch_stable.html

#download data for prepare_data.py
#sources
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1hPThYJKlyLFLqZiQzW0hAOXAcz1pydg_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hPThYJKlyLFLqZiQzW0hAOXAcz1pydg_" -O wikiauto_sources.pickle && rm -rf /tmp/cookies.txt
#targets
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1E5ty4BjIjyvIvimMkqqJwULtLLofw4IN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E5ty4BjIjyvIvimMkqqJwULtLLofw4IN" -O wikiauto_targets.pickle && rm -rf /tmp/cookies.txt
#aug_data
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1vxWDSs4Y52l-EUoojRwvIBpbvLuWJjfc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vxWDSs4Y52l-EUoojRwvIBpbvLuWJjfc" -O augmented_wikiauto_max_cnt_2.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=10oFocS84miPBOWXh5YrVYYRS8u2wkZvh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10oFocS84miPBOWXh5YrVYYRS8u2wkZvh" -O augmented_wikiauto_max_cnt_6.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1woOcQgNXKGSOrOkOte23YFAOEC1zpyLj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1woOcQgNXKGSOrOkOte23YFAOEC1zpyLj" -O augmented_wikiauto_max_cnt_14.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ffEP26UINC-5sWXxcJ5TfCdCERl7Upzv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ffEP26UINC-5sWXxcJ5TfCdCERl7Upzv" -O augmented_wikiauto_max_cnt_30.pickle && rm -rf /tmp/cookies.txt

#finetuned checkpoints
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Lytj61Rso_kR8jAU8YtT0-136hmSLCYM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Lytj61Rso_kR8jAU8YtT0-136hmSLCYM" -O epoch=9.ckpt && rm -rf /tmp/cookies.txt

#random sampled data all
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1QLPDxLQBZsY0-zWoRrH7HHL4BedvfcMM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QLPDxLQBZsY0-zWoRrH7HHL4BedvfcMM" -O augmented_wikiauto_max_cnt_6_randomsamp_16_labeled_by_exp_1.pickle && rm -rf /tmp/cookies.txt
#random sampled data filtered
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1sQlVIUqivMuXcIeEJYTvo4oPCaWke4QJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sQlVIUqivMuXcIeEJYTvo4oPCaWke4QJ" -O filtered_augmented_wikiauto_max_cnt_6_randomsamp_16_labeled_by_exp_1.pickle && rm -rf /tmp/cookies.txt
#random sampled data labeled
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1c-7hSiGQLrbgOq9yqXs-EDQ4dDIZ6pvL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c-7hSiGQLrbgOq9yqXs-EDQ4dDIZ6pvL" -O random_sampled_df_labeled_max_cnt_6_randomsamp_16.pickle && rm -rf /tmp/cookies.txt


#wikiauto_dataframe
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AEFplcg5_lE5plDKzoLzXEL3GqB-q23d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AEFplcg5_lE5plDKzoLzXEL3GqB-q23d" -O wikiauto_dataframe.pickle && rm -rf /tmp/cookies.txt
#wikiauto_dataframe_addfeatures
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ZAgWqpol18OqVIt5sX5eW_XOsrpGqnVc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZAgWqpol18OqVIt5sX5eW_XOsrpGqnVc" -O wikiauto_dataframe_addfeatures.pickle && rm -rf /tmp/cookies.txt

#move files
mv w* ../../data
mv a* ../../data
mv epoch=9.ckpt ../../data
mv f* ../../data
mv r* ../../data

#exec prepare_data.py
#python ../../data/prepare_data.py

