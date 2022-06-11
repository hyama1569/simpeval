import pickle
import json
import pandas as pd

def reverse_orig_simp(orig_sents, simp_sents):
    df_rev = pd.DataFrame({'original':simp_sents, 'simple':orig_sents})
    df_rev['label'] = 0
    return df_rev

if __name__ == '__main__':
    with open('./src/wiki-auto-part-1-data.json?dl=0', 'r') as f:
        json_dict = json.load(f)

    sources = []
    targets = []

    for v in json_dict.values():
        sentence_alignment = v['sentence_alignment']
        for i in range(len(sentence_alignment)):
            simp_key = sentence_alignment[i][0]
            norm_key = sentence_alignment[i][1]
            sources.append(v['normal']['content'][norm_key])
            targets.append(v['simple']['content'][simp_key])

    with open('./src/wikiauto_sources.pickle', 'wb') as f:
        pickle.dump(sources, f)
    with open('./src/wikiauto_targets.pickle', 'wb') as f:
        pickle.dump(targets, f)

    df = pd.DataFrame({'original':sources, 'simple':targets})
    df['label'] = 1
    df_rev = reverse_orig_simp(sources, targets)
    data = pd.concat([df, df_rev], axis=0).reset_index()

    with open('./src/wikiauto_dataframe.pickle', 'wb') as f:
        pickle.dump(data, f)