from utils_EditNTS import *
from utils_neural_jacana import *
from utils_extract_edit_operation import *
from utils_apply_operations import *
import tqdm
import spacy
import pickle

if __name__ == '__main__':
    #with open('./src/wikiauto_sources.pickle', 'rb') as f:
    #    sources = pickle.load(f)
    #with open('./src/wikiauto_targets.pickle', 'rb') as f:
    #    targets = pickle.load(f)
    #with open('./src/aligns_wikiauto.pickle', 'rb') as f:
    #    aligns = pickle.load(f)

    with open('./src/wikiauto_dataframe_addfeatures.pickle', 'rb') as f:
        data = pickle.load(f)

    sources = data["original"].tolist()
    targets = data["simple"].tolist()
    edit_sequences = data["edit_sequences"].tolist()
        
    nltk.download('punkt')
    sent1_toks = preprocess_texts(sources)
    sent2_toks = preprocess_texts(targets)
    nlp = spacy.load('en_core_web_sm')

    #edit_sequences = get_edit_sequences(sent1_toks, sent2_toks, aligns)
    max_cnt = 2
    #max_cnt = 4 #6-2=4
    #max_cnt = 8 #14-(2+4)=8 
    applied_sentences_all, applied_edit_sequences_all = apply_edit_sequences(edit_sequences, sent1_toks, sent2_toks, nlp, max_cnt)
    with open('./src/augmented_wikiauto_max_cnt_2.pickle', 'wb') as f:
        pickle.dump(applied_sentences_all, f)
    with open('./src/augmented_wikiauto_applied_sequences_max_cnt_2.pickle', 'wb') as f:
        pickle.dump(applied_edit_sequences_all, f)
