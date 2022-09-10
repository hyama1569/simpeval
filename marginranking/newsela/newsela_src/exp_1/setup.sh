#!/bin/bash

#requirements install
pip install wandb
wandb login
pip install git+https://github.com/facebookresearch/text-simplification-evaluation.git
pip install transformers
#pip install -U pip setuptools wheel
#pip install setuptools==59.5.0
#pip install -U spacy
pip install git+https://github.com/PyTorchLightning/pytorch-lightning
pip install hydra-core --upgrade
python -m spacy download en_core_web_sm
pip install torch==1.9+cu111  -f https://download.pytorch.org/whl/torch_stable.html

##newsela_dataframe_addfeatures
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1DaIjerq6PM6fD6qWllgbWWkPeD3wdNoT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DaIjerq6PM6fD6qWllgbWWkPeD3wdNoT" -O newsela_dataframe_addfeatures_100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-0-SczrFGF24LA9WS-79HkHaDbeZJmYk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0-SczrFGF24LA9WS-79HkHaDbeZJmYk" -O newsela_dataframe_addfeatures_75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-2BeMNUvcWDGly5J8t6AcOLsxTSRoHLE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-2BeMNUvcWDGly5J8t6AcOLsxTSRoHLE" -O newsela_dataframe_addfeatures_50.pickle && rm -rf /tmp/cookies.txt
#25
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-4ZlBfvrbUyC3_4GcDB2tq7quikgv5N_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-4ZlBfvrbUyC3_4GcDB2tq7quikgv5N_" -O newsela_dataframe_addfeatures_25.pickle && rm -rf /tmp/cookies.txt

##newsela_dataframe_addfeatures_max_cnt_2
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-CraE1ua2AGa-NPwBaSCPFmQ_Su8J0xT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-CraE1ua2AGa-NPwBaSCPFmQ_Su8J0xT" -O newsela_dataframe_addfeatures_max_cnt__100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-HnyLVqixHi2ymv40eTUCTk8Nx4KvNFk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-HnyLVqixHi2ymv40eTUCTk8Nx4KvNFk" -O newsela_dataframe_addfeatures_max_cnt__75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-NzbECifDybKBDA8uAskFdKWrT_GuOXZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-NzbECifDybKBDA8uAskFdKWrT_GuOXZ" -O newsela_dataframe_addfeatures_max_cnt__50.pickle && rm -rf /tmp/cookies.txt
#25
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-QDsrZtJWT85mxgYjZ3WuM8LKRXB5gRg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-QDsrZtJWT85mxgYjZ3WuM8LKRXB5gRg" -O newsela_dataframe_addfeatures_max_cnt__25.pickle && rm -rf /tmp/cookies.txt

##newsela_dataframe_addfeatures_max_cnt_3
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-R_ZHOqTyNCG-D8fzRg04t9VyAwL27Gx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-R_ZHOqTyNCG-D8fzRg04t9VyAwL27Gx" -O newsela_dataframe_addfeatures_max_cnt_3_100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-fa4aJFatog02s8azoyabTUpyoj7Y4mc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-fa4aJFatog02s8azoyabTUpyoj7Y4mc" -O newsela_dataframe_addfeatures_max_cnt_3_75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-fjcTJiZlm9vDsri8Xfa8kCd7wKQmLRG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-fjcTJiZlm9vDsri8Xfa8kCd7wKQmLRG" -O newsela_dataframe_addfeatures_max_cnt_3_50.pickle && rm -rf /tmp/cookies.txt
#25
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-h-0bYi4qr-SSRMjqJJvGsSC8cf-PkWg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-h-0bYi4qr-SSRMjqJJvGsSC8cf-PkWg" -O newsela_dataframe_addfeatures_max_cnt_3_25.pickle && rm -rf /tmp/cookies.txt

#move files
mv n* ../../data