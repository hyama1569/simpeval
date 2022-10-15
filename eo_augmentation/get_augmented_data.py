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

    #with open('./src/wikiauto_dataframe_addfeatures.pickle', 'rb') as f:
    #    data = pickle.load(f)

    with open('./src/newsela_dataframe.pickle', 'rb') as f:
        data = pickle.load(f)

    random_seed = 1
    newsela_3more_100 = data[data['comp_simp_diff'] >= 3]
    newsela_3more_75 = data[data['comp_simp_diff'] >= 3].sample(n=int(3*len(data[data['comp_simp_diff'] >= 3])/4), random_state=random_seed)
    newsela_3more_50 = newsela_3more_75.sample(n=int(2*len(newsela_3more_75)/3), random_state=random_seed)
    newsela_3more_25 = newsela_3more_50.sample(n=int(len(newsela_3more_50)/2), random_state=random_seed)

    sources = newsela_3more_25["original"].tolist()
    targets = newsela_3more_25["simple"].tolist()
    edit_sequences = newsela_3more_25["edit_sequences"].tolist()

        
    nltk.download('punkt')
    sent1_toks = preprocess_texts(sources)
    sent2_toks = preprocess_texts(targets)
    nlp = spacy.load('en_core_web_sm')

    #edit_sequences = get_edit_sequences(sent1_toks, sent2_toks, aligns)
    max_cnt = 2
    #max_cnt = 4 #6-2=4
    #max_cnt = 8 #14-(2+4)=8 
    applied_sentences_all, applied_edit_sequences_all = apply_edit_sequences(edit_sequences, sent1_toks, sent2_toks, nlp, max_cnt)
    #applied_sentences_all, applied_edit_sequences_all = apply_edit_sequences_sampling_without_replacement(edit_sequences, sent1_toks, sent2_toks, nlp)
    with open('./src/augmented_newsela_3more_25.pickle', 'wb') as f:
        pickle.dump(applied_sentences_all, f)
    with open('./src/augmented_newsela_3more_applied_sequences_25.pickle', 'wb') as f:
        pickle.dump(applied_edit_sequences_all, f)
