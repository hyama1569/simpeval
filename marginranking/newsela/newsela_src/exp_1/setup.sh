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
#pip install torch==1.9+cu111  -f https://download.pytorch.org/whl/torch_stable.html

##newsela_dataframe_addfeatures
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CRi14wp_LYn0_8pdzF3Rdc_SZpdIiGsp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CRi14wp_LYn0_8pdzF3Rdc_SZpdIiGsp" -O newsela_3more_dataframe_addfeatures_100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-6lwZRBMazcTzxO0pSwyYhlTZdhWwMUU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6lwZRBMazcTzxO0pSwyYhlTZdhWwMUU" -O newsela_3more_dataframe_addfeatures_75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-CcMIlPD08XccIzMl8njYaP1R5aCAz-_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-CcMIlPD08XccIzMl8njYaP1R5aCAz-_" -O newsela_3more_dataframe_addfeatures_50.pickle && rm -rf /tmp/cookies.txt
#25
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-IRjO93Z2qlbskiO_swjqsktjtnTOnSj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-IRjO93Z2qlbskiO_swjqsktjtnTOnSj" -O newsela_3more_dataframe_addfeatures_25.pickle && rm -rf /tmp/cookies.txt

##newsela_dataframe_addfeatures_max_cnt_2
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-Ko9gc_R1pOYuU9v9CEPie24YJC_uk56' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Ko9gc_R1pOYuU9v9CEPie24YJC_uk56" -O newsela_3more_dataframe_addfeatures_max_cnt_2_100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-Nq2We3jNN8YlRUzKvuTUokPQDzBFQlZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Nq2We3jNN8YlRUzKvuTUokPQDzBFQlZ" -O newsela_3more_dataframe_addfeatures_max_cnt_2_75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-OvHYB0MzQboJnV0HSHB_lk6oQygQDyL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-OvHYB0MzQboJnV0HSHB_lk6oQygQDyL" -O newsela_3more_dataframe_addfeatures_max_cnt_2_50.pickle && rm -rf /tmp/cookies.txt
#25
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-OxK5BV1TSh-1B_ZKKFaR7cP8vLzKtDz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-OxK5BV1TSh-1B_ZKKFaR7cP8vLzKtDz" -O newsela_3more_dataframe_addfeatures_max_cnt_2_25.pickle && rm -rf /tmp/cookies.txt

##newsela_dataframe_addfeatures_max_cnt_3
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-Sya_E2bqm1bL3_ELbq5OXpTSyqM-6BN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Sya_E2bqm1bL3_ELbq5OXpTSyqM-6BN" -O newsela_3more_dataframe_addfeatures_max_cnt_3_100.pickle && rm -rf /tmp/cookies.txt
#75
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-TgCYsf75W_rp-kA47l5hOde3zAEOtmX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-TgCYsf75W_rp-kA47l5hOde3zAEOtmX" -O newsela_3more_dataframe_addfeatures_max_cnt_3_75.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-Xs_jKMTbiO0j_n8GS1vBJYwMMZBLOKp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Xs_jKMTbiO0j_n8GS1vBJYwMMZBLOKp" -O newsela_3more_dataframe_addfeatures_max_cnt_3_50.pickle && rm -rf /tmp/cookies.txt
#25
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-ZK06zCIGH8_ZlkMZ3u0X_r5QdU6li6-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-ZK06zCIGH8_ZlkMZ3u0X_r5QdU6li6-" -O newsela_3more_dataframe_addfeatures_max_cnt_3_25.pickle && rm -rf /tmp/cookies.txt

##newsela_only_addfeatures
#100
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1--VMqPxqUwAg6RJBQpFotQe07GkoEIeX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--VMqPxqUwAg6RJBQpFotQe07GkoEIeX" -O newsela_only_addfeatures_100.pickle && rm -rf /tmp/cookies.txt
#50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1--MPOBYVkfcacsy6luIXGhHtgi5SEnli' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--MPOBYVkfcacsy6luIXGhHtgi5SEnli" -O newsela_only_addfeatures_50.pickle && rm -rf /tmp/cookies.txt
#no_aug
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1zYNv9Kp_6uHA8wErXbwkmLeWZcKYvayY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zYNv9Kp_6uHA8wErXbwkmLeWZcKYvayY" -O newsela_only_addfeatures_no_aug.pickle && rm -rf /tmp/cookies.txt

##newsela_dataframe_swor_addfeatures
#100
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-D8ST-OSumeREfGX8FIz2hvhHAGB_lAX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-D8ST-OSumeREfGX8FIz2hvhHAGB_lAX" -O newsela_3more_dataframe_swor_addfeatures_100.pickle && rm -rf /tmp/cookies.txt
#75
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-CZwvgi09qk5sJUWT-zllEUsGBJV1lK0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-CZwvgi09qk5sJUWT-zllEUsGBJV1lK0" -O newsela_3more_dataframe_swor_addfeatures_75.pickle && rm -rf /tmp/cookies.txt
#50
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-5MMiHpicz6d09QT0YZwZqgnG7g9PhOV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-5MMiHpicz6d09QT0YZwZqgnG7g9PhOV" -O newsela_3more_dataframe_swor_addfeatures_50.pickle && rm -rf /tmp/cookies.txt
#25
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1zog4clGr94aZBIZnPd6ksPjWBsSoLvGC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zog4clGr94aZBIZnPd6ksPjWBsSoLvGC" -O newsela_3more_dataframe_swor_addfeatures_25.pickle && rm -rf /tmp/cookies.txt


#move files
mv n* ../../data