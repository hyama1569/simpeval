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

#exp_1
##25%
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1qCASeu22vwFYuR_BA3BMZeGkG18EuDYT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qCASeu22vwFYuR_BA3BMZeGkG18EuDYT" -O newsela_3more_dataframe_ext1_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1aUZys62aVPUnuIgqjimetpeLWC3Cmtom' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aUZys62aVPUnuIgqjimetpeLWC3Cmtom" -O newsela_3more_dataframe_ext2_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xt0DSqDSNNkss7sih4zd8WcyvAF-biYb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xt0DSqDSNNkss7sih4zd8WcyvAF-biYb" -O newsela_3more_dataframe_ext3_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1cJZE8X0CI5GH4iypP2IzA7wiJREDH28b' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cJZE8X0CI5GH4iypP2IzA7wiJREDH28b" -O newsela_3more_dataframe_ext4_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LCZRwMHpeAdqSmbwLcjkln5gors5fDaU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LCZRwMHpeAdqSmbwLcjkln5gors5fDaU" -O newsela_3more_dataframe_ext5_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1BTrKSf9BH3uK7j9JlOEyL6kN58BnUONo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BTrKSf9BH3uK7j9JlOEyL6kN58BnUONo" -O newsela_3more_dataframe_ext6_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Mit3zAvmgKMdQZ7hMBz7UHH19TUfv89v' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Mit3zAvmgKMdQZ7hMBz7UHH19TUfv89v" -O newsela_3more_dataframe_ext7_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1qtp3JQHSdlIs4i4KFw9ThWvMY45jV-Uy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qtp3JQHSdlIs4i4KFw9ThWvMY45jV-Uy" -O newsela_3more_dataframe_ext8_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GFicFloR3lNvu4QAQ0BsubVR-Aufz-bt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GFicFloR3lNvu4QAQ0BsubVR-Aufz-bt" -O newsela_3more_dataframe_ext9_25.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1pgD0tln86yi2VPi8ScXHdFos20nB2T7L' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pgD0tln86yi2VPi8ScXHdFos20nB2T7L" -O newsela_3more_dataframe_ext10_25.pickle && rm -rf /tmp/cookies.txt
##50%
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JAC2dx9wzj_bb5ego3SBAbdsV1Rltn1a' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JAC2dx9wzj_bb5ego3SBAbdsV1Rltn1a" -O newsela_3more_dataframe_ext1_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1K0_cTnnUn8wrlLxS5ih-zsp5uwns7C-6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K0_cTnnUn8wrlLxS5ih-zsp5uwns7C-6" -O newsela_3more_dataframe_ext2_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=163fd6RcRhU_1R0V_rqQg8llO8-8qDEil' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=163fd6RcRhU_1R0V_rqQg8llO8-8qDEil" -O newsela_3more_dataframe_ext3_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1WJsLgv39eks25GTEAKR3QykikdplMS76' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WJsLgv39eks25GTEAKR3QykikdplMS76" -O newsela_3more_dataframe_ext4_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1KgihNSLUum0vKawYbOZWBQeRzcvvL-mD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KgihNSLUum0vKawYbOZWBQeRzcvvL-mD" -O newsela_3more_dataframe_ext5_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1gQ0rQ8gT9E6uEJ5kZq6j54UYL9nN8p79' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gQ0rQ8gT9E6uEJ5kZq6j54UYL9nN8p79" -O newsela_3more_dataframe_ext6_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1wBlB05dIvoOLPjpC4uW_hmEibEpCJXEj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wBlB05dIvoOLPjpC4uW_hmEibEpCJXEj" -O newsela_3more_dataframe_ext7_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1nEGTAPhzG0tqVR_CfyOOZ4RLPq3eBRjI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nEGTAPhzG0tqVR_CfyOOZ4RLPq3eBRjI" -O newsela_3more_dataframe_ext8_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1N_k0cg_rKyGS3SlXkyaK-jvmnOXgYd8y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N_k0cg_rKyGS3SlXkyaK-jvmnOXgYd8y" -O newsela_3more_dataframe_ext9_50.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1DEbCD88JPCfGl_mWpQuBqCqwPJTmUxjG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DEbCD88JPCfGl_mWpQuBqCqwPJTmUxjG" -O newsela_3more_dataframe_ext10_50.pickle && rm -rf /tmp/cookies.txt
##75%
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1g7SoqBlbHM-n9XikrTG-CdsLltHM7jli' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g7SoqBlbHM-n9XikrTG-CdsLltHM7jli" -O newsela_3more_dataframe_ext1_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1n_5uVohptQOOOQ9jqJJq117IbmbvoN0q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n_5uVohptQOOOQ9jqJJq117IbmbvoN0q" -O newsela_3more_dataframe_ext2_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1w-9-fbjv6m6uiuvjESidK8HkaiTEHoKc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1w-9-fbjv6m6uiuvjESidK8HkaiTEHoKc" -O newsela_3more_dataframe_ext3_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1b4SWoT5_UxhZ68iz_DlsFmkIDLWwelLQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b4SWoT5_UxhZ68iz_DlsFmkIDLWwelLQ" -O newsela_3more_dataframe_ext4_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=18IqkquT25lEnc_rK3L6y-RTah6hiQ0ra' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18IqkquT25lEnc_rK3L6y-RTah6hiQ0ra" -O newsela_3more_dataframe_ext5_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=11oa1R0JYUrooUwp17KO5YuKABgP8hpeb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11oa1R0JYUrooUwp17KO5YuKABgP8hpeb" -O newsela_3more_dataframe_ext6_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1DCLynS87IfhjzXmhBompfFxWAbz0ITKC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DCLynS87IfhjzXmhBompfFxWAbz0ITKC" -O newsela_3more_dataframe_ext7_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1WgyrLUyPL5B6Sgzfvt5EcpXGZKAB3Dt5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WgyrLUyPL5B6Sgzfvt5EcpXGZKAB3Dt5" -O newsela_3more_dataframe_ext8_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AzwtgEcqrstXrV51bqgfQz68Dgsgu-AJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AzwtgEcqrstXrV51bqgfQz68Dgsgu-AJ" -O newsela_3more_dataframe_ext9_75.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1TAKe99e-Kv6qc4G8BUl_-tdX17oZQuWY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TAKe99e-Kv6qc4G8BUl_-tdX17oZQuWY" -O newsela_3more_dataframe_ext10_75.pickle && rm -rf /tmp/cookies.txt
##100%
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ctiVWwGscQ6swBLFi9NXH2mgxzfM7tBM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ctiVWwGscQ6swBLFi9NXH2mgxzfM7tBM" -O newsela_3more_dataframe_100.pickle && rm -rf /tmp/cookies.txt


#exp_2 (50%)
##aug_50
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=11bUWpe2zRVzdsKMcij9DKbHrnXqkPSNd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11bUWpe2zRVzdsKMcij9DKbHrnXqkPSNd" -O newsela_3more_dataframe_ext11_eo11_ag11_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Wv4JysYqNM4Te6MAUql7lFTV4ZJyxQXb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Wv4JysYqNM4Te6MAUql7lFTV4ZJyxQXb" -O newsela_3more_dataframe_ext12_eo12_ag12_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1h5-o9bwoylZr_Es_8ayUXdCm5cl0W0pQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h5-o9bwoylZr_Es_8ayUXdCm5cl0W0pQ" -O newsela_3more_dataframe_ext13_eo13_ag13_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Dq3XBdyfJSEx4Aj526m0Ab69-6n2A2VN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Dq3XBdyfJSEx4Aj526m0Ab69-6n2A2VN" -O newsela_3more_dataframe_ext14_eo14_ag14_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ukHDz1ZteL1iwi5SkCbp_FfEeE_uKUCe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ukHDz1ZteL1iwi5SkCbp_FfEeE_uKUCe" -O newsela_3more_dataframe_ext15_eo15_ag15_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1X5HbovRwJoFmC5hPsGRbNGdxzqMXz8rv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X5HbovRwJoFmC5hPsGRbNGdxzqMXz8rv" -O newsela_3more_dataframe_ext16_eo16_ag16_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=19VtoxMQnrMGx82Cgj46bVLIok9OBumRT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19VtoxMQnrMGx82Cgj46bVLIok9OBumRT" -O newsela_3more_dataframe_ext17_eo17_ag17_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CQkjfhlTA9TepxR3swSuYZAQddFihO4Z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CQkjfhlTA9TepxR3swSuYZAQddFihO4Z" -O newsela_3more_dataframe_ext18_eo18_ag18_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XczbQUuPtqqHFMHq4dngQ1atM2lWxsrV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XczbQUuPtqqHFMHq4dngQ1atM2lWxsrV" -O newsela_3more_dataframe_ext19_eo19_ag19_aug50_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1O8FrUWAkQTHkX6aSyP1twFUmmVxPFip0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1O8FrUWAkQTHkX6aSyP1twFUmmVxPFip0" -O newsela_3more_dataframe_ext20_eo20_ag20_aug50_50.pickle && rm -rf /tmp/cookies.txt
##aug_100
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ctujCq-Ci8V8c5v56qssNQx9NrjtuyTj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ctujCq-Ci8V8c5v56qssNQx9NrjtuyTj" -O newsela_3more_dataframe_ext11_eo11_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1BGMyBl-kkaEZYhCF2HrlqERsvV2kb3Zy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BGMyBl-kkaEZYhCF2HrlqERsvV2kb3Zy" -O newsela_3more_dataframe_ext12_eo12_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1933e58DCzbtHEXOy69nIPmM-AilSyJKt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1933e58DCzbtHEXOy69nIPmM-AilSyJKt" -O newsela_3more_dataframe_ext13_eo13_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1jDS9KksE3fdbc04wqdewbt0Ytw130QJ5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jDS9KksE3fdbc04wqdewbt0Ytw130QJ5" -O newsela_3more_dataframe_ext14_eo14_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LQbzSnrWTlbv9iEKmDstFgRdtBh2Mu5y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LQbzSnrWTlbv9iEKmDstFgRdtBh2Mu5y" -O newsela_3more_dataframe_ext15_eo15_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Hsa2q10_0GYL9VKfFGnajFxmujg27uIg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Hsa2q10_0GYL9VKfFGnajFxmujg27uIg" -O newsela_3more_dataframe_ext16_eo16_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1BPptHVbZlbfQ46W30w5VhHXWAAxBEAU-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BPptHVbZlbfQ46W30w5VhHXWAAxBEAU-" -O newsela_3more_dataframe_ext17_eo17_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1y0O49_mlNg_Whppcpyy8A7TZ_FHQqOz-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1y0O49_mlNg_Whppcpyy8A7TZ_FHQqOz-" -O newsela_3more_dataframe_ext18_eo18_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fZ3x8hLMl9e9Bd0DiL5ZlvagOZ1IipK4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fZ3x8hLMl9e9Bd0DiL5ZlvagOZ1IipK4" -O newsela_3more_dataframe_ext19_eo19_aug100_50.pickle && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1v1N2Y8YfpGCDwubfBF4j3__Q9SGfS70c' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v1N2Y8YfpGCDwubfBF4j3__Q9SGfS70c" -O newsela_3more_dataframe_ext20_eo20_aug100_50.pickle && rm -rf /tmp/cookies.txt


#exp_2 (100%)
##aug_50
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1NwGNl0RBCFWFvXJR_vuwR7zyHnODN9SV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NwGNl0RBCFWFvXJR_vuwR7zyHnODN9SV" -O newsela_3more_dataframe_eo1_ag1_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1bxgYNirPHTz_uaBxS8QOz_p5v2GyDRu9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bxgYNirPHTz_uaBxS8QOz_p5v2GyDRu9" -O newsela_3more_dataframe_eo2_ag2_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1tDI6X6Wp6_NOIVwNq64H2dqejVXCuxjd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tDI6X6Wp6_NOIVwNq64H2dqejVXCuxjd" -O newsela_3more_dataframe_eo3_ag3_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=19GUQh0Z0_RZF_FBUAIig1cup9sQ-Ucpa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19GUQh0Z0_RZF_FBUAIig1cup9sQ-Ucpa" -O newsela_3more_dataframe_eo4_ag4_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1a5ilQ_quQKF2kt3Frp0mAAQdU0K80Bv5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a5ilQ_quQKF2kt3Frp0mAAQdU0K80Bv5" -O newsela_3more_dataframe_eo5_ag5_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Q1jSVxj02qM5XOue2biWji3s7TsB_mi2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Q1jSVxj02qM5XOue2biWji3s7TsB_mi2" -O newsela_3more_dataframe_eo6_ag6_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fPU-R0Lii-vJeK7K0i3lJf6sHdg3_Eox' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fPU-R0Lii-vJeK7K0i3lJf6sHdg3_Eox" -O newsela_3more_dataframe_eo7_ag7_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fh1lYyzzk9NdtZ6Emkq2T9yIQPiQHYDX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fh1lYyzzk9NdtZ6Emkq2T9yIQPiQHYDX" -O newsela_3more_dataframe_eo8_ag8_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1NM0Lr-p_qkH4-8jPM8bn-SRqO7VuUZN4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NM0Lr-p_qkH4-8jPM8bn-SRqO7VuUZN4" -O newsela_3more_dataframe_eo9_ag9_aug50_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=11sId8JY4Vspfk3ckuW-nRFHcQrDTSdjV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11sId8JY4Vspfk3ckuW-nRFHcQrDTSdjV" -O newsela_3more_dataframe_eo10_ag10_aug50_100.pickle && rm -rf /tmp/cookies.txt
##aug_100 
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-n8Kho6ahS82TlnRCPbJcl9x2NWO4aRu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-n8Kho6ahS82TlnRCPbJcl9x2NWO4aRu" -O newsela_3more_dataframe_eo1_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CfT5NcVpLn5toKj1GsfuwX1CDT2mvpD8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CfT5NcVpLn5toKj1GsfuwX1CDT2mvpD8" -O newsela_3more_dataframe_eo2_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1R91qXRN9eCCBhZItRvsg3N79jX33fMEK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R91qXRN9eCCBhZItRvsg3N79jX33fMEK" -O newsela_3more_dataframe_eo3_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=15sm_Z_L_-ppgtf6MOZ42DeuG1jx_3rIk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15sm_Z_L_-ppgtf6MOZ42DeuG1jx_3rIk" -O newsela_3more_dataframe_eo4_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Z9S3ppWGAehgFTNsgLOI4k_uPrOdBnQc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z9S3ppWGAehgFTNsgLOI4k_uPrOdBnQc" -O newsela_3more_dataframe_eo5_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1aQh_L80BpvhAVHkQBRishY_0HfYCdlBt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aQh_L80BpvhAVHkQBRishY_0HfYCdlBt" -O newsela_3more_dataframe_eo6_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1t-ZvxRPdD_0WT4GHF2HWaypb5ZUuLkG6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1t-ZvxRPdD_0WT4GHF2HWaypb5ZUuLkG6" -O newsela_3more_dataframe_eo7_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1rSKlqCzdF_fMNJkwosyiua5Mx1CkKeN-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rSKlqCzdF_fMNJkwosyiua5Mx1CkKeN-" -O newsela_3more_dataframe_eo8_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1i661jDMWHiSaQD-QHMOTWvV5pNcXJ4FA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i661jDMWHiSaQD-QHMOTWvV5pNcXJ4FA" -O newsela_3more_dataframe_eo9_aug100_100.pickle && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AOLPG45Z6exhKpgInV3Jok8IcjrxaX25' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AOLPG45Z6exhKpgInV3Jok8IcjrxaX25" -O newsela_3more_dataframe_eo10_aug100_100.pickle && rm -rf /tmp/cookies.txt

#move files
mv n* ../../data