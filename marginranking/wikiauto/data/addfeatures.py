import nltk
nltk.download('punkt')
import tqdm
from tseval.feature_extraction import get_compression_ratio, get_wordrank_score
import spacy
import pickle

class AdditionalFeatureExtractor():
    def __init__(
        self,
        origin: str,
        sent: str,
        edit_sequences: object,
    ):
        self.origin = origin
        self.sent = sent
        self.edit_sequences = edit_sequences
    
    def wordrank_score(self):
        return get_wordrank_score(self.sent)
    
    def maximum_deptree_depth(self):
        def tree_height(root):
            if not list(root.children):
                return 1
            else:
                return 1 + max(tree_height(x) for x in root.children)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.sent)
        roots = [sent.root for sent in doc.sents]
        return max([tree_height(root) for root in roots])

    def compression_ratio(self):
        return get_compression_ratio(self.origin, self.sent)
    
    def edit_operation_nums(self):
        return len(self.edit_sequences)

with open('./wikiauto_dataframe_addfeatures.pickle', 'rb') as f:
    data = pickle.load(f)

wordrank_score = []
wordrank_score_orig = []
max_dep_depth = []
max_dep_depth_orig = []
comp_ratio = []
edit_sequences_len = []
for i in tqdm.tqdm(range(len(data))):
    origin = data.iloc[i]["origin"]
    sent = data.iloc[i]["simple"]
    edit_sequences = data.iloc[i]["edit_sequences"]
    fe = AdditionalFeatureExtractor(origin=origin, sent=sent, edit_sequences=edit_sequences)
    wordrank_score.append(fe.wordrank_score())
    max_dep_depth.append(fe.maximum_deptree_depth())
    comp_ratio.append(fe.compression_ratio())
    edit_sequences_len.append(fe.edit_operation_nums())

    fe_orig = AdditionalFeatureExtractor(origin=origin, sent=origin, edit_sequences=edit_sequences)
    wordrank_score_orig.append(fe_orig.wordrank_score())
    max_dep_depth_orig.append(fe_orig.maximum_deptree_depth())

data["wordrank_score"] = wordrank_score
data["wordrank_score_orig"] = wordrank_score_orig

data["max_dep_depth"] = max_dep_depth
data["max_dep_depth_orig"] = max_dep_depth_orig

data["comp_ratio"] = comp_ratio
data["edit_sequences_len"] = edit_sequences_len

with open('./wikiauto_dataframe_addfeatures.pickle', 'wb') as f:
    pickle.dump(f)