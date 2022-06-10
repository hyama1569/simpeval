from utils_EditNTS import *
from utils_neural_jacana import *
from utils_extract_edit_operation import *
from utils_apply_operations import *
import tqdm
import spacy
import pickle

if __name__ == '__main__':
    with open('./src/wikiauto_sources.pickle', 'rb') as f:
        sources = pickle.load(f)
    with open('./src/wikiauto_targets.pickle', 'rb') as f:
        targets = pickle.load(f)
    with open('./src/aligns_wikiauto.pickle', 'rb') as f:
        aligns = pickle.load(f)

    sent1_toks = preprocess_texts(sources)
    sent2_toks = preprocess_texts(targets)
    nlp = spacy.load('en_core_web_sm')

    edit_sequences = get_edit_sequences(sent1_toks, sent2_toks, aligns)
    max_cnt = 10
    applied_sentences_all = apply_edit_sequences(edit_sequences, sent1_toks, sent2_toks, nlp, max_cnt)
    with open('./src/augmented_wikiauto.pickle', 'wb') as f:
                pickle.dump(applied_sentences_all, f)