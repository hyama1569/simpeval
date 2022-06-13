#!/bin/bash

pip install transformers
pip install -U spacy
python -m spacy download en_core_web_sm

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JS7UeElzTGr0eQZ-H_556CiQif15LuEa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JS7UeElzTGr0eQZ-H_556CiQif15LuEa" -O aligns_wikiauto.pickle && rm -rf /tmp/cookies.txt
