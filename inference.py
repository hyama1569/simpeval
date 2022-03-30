import argparse
from transformers import BertTokenizer
from collections import namedtuple
import random
import torch
import numpy as np
import re
import nltk
from .neural_jacana.model import *

def preprocess_texts(texts):
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(nltk.word_tokenize(text))
    return tokenized_texts

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def merge_sent2_ids(allign_id_pairs):
    '''
    merge sent2 ids alligned to sent1 ids.
    '''
    merged_sent2_allign_id_pairs = []
    tuple_pairs = [(int(re.findall(r"\d+", i)[0]), int(re.findall(r"\d+", i)[1])) for i in sorted(allign_id_pairs, key=lambda x:(int(re.findall(r"\d+", x)[0]), int(re.findall(r"\d+", x)[1])))]
    for pair in tuple_pairs:
        keys = [i[0] for i in merged_sent2_allign_id_pairs]
        keys_flatten = [x for row in keys for x in row]
        if pair[0] not in keys_flatten:
            merged_sent2_allign_id_pairs.append(([pair[0]], [pair[1]]))
        else:
            ind_addval = [i for i in range(len(keys)) if pair[0] in keys[i][0]]
            merged_sent2_allign_id_pairs[ind_addval][1].append(pair[1])
    return merged_sent2_allign_id_pairs

def merge_sent1_ids(merged_sent2_allign_id_pairs):
    '''
    merge sent1 ids having the same sent2 ids.
    '''
    merged_sent1_allign_id_pairs = []
    dup_inds = []
    vals = [pair[1] for pair in merged_sent2_allign_id_pairs]
    for pair in merged_sent2_allign_id_pairs:
        dup_ind = [i for i, x in enumerate(vals) if x == pair[1]]
        if len(dup_ind) > 1:
            dup_inds.append(dup_ind)
    dup_inds = get_unique_list(dup_inds)

    pair_to_add = []
    for i in range(len(merged_sent2_allign_id_pairs)):
        for j in range(len(dup_inds)):
            if merged_sent2_allign_id_pairs[i][0][0] in dup_inds[j]:
                pair_to_add.append((dup_inds[j], merged_sent2_allign_id_pairs[i][1]))
    pair_to_add = get_unique_list(pair_to_add)

    dup_inds_flatten = [x for row in dup_inds for x in row]
    for i in range(len(merged_sent2_allign_id_pairs)):
        if i not in dup_inds_flatten:
            merged_sent1_allign_id_pairs.append(merged_sent2_allign_id_pairs[i])
    merged_sent1_allign_id_pairs.extend(pair_to_add)
    return merged_sent1_allign_id_pairs

def ids_to_words(merged_sent1_allign_id_pairs, tokenized_sent1, tokenized_sent2):
    allign_word_pairs = []
    for pair in merged_sent1_allign_id_pairs:
        sent1_words = [tokenized_sent1[i] for i in pair[0]]
        sent2_words = [tokenized_sent2[i] for i in pair[1]]
        allign_id_pairs.append((sent1_words, sent2_words))
    return allign_word_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_epoch", default=6, type=int)
    parser.add_argument("--max_span_size", default=4, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_sent_length", default=70, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--dataset", default='mtref', type=str)
    parser.add_argument("--sure_and_possible", default='True', type=str)
    parser.add_argument("--distance_embedding_size", default=128, type=int)
    parser.add_argument("--use_transition_layer", default='False', type=str, help='if False, will set transition score to 0.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = NeuralWordAligner(args)
    my_device = torch.device('cpu')
    model = model.to(my_device)

    checkpoint = torch.load('./neural_jacana/Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt', map_location=my_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sources = ["Military experts say the line between combat is getting blurry.", "Their eyes are quite small, and their visual acuity is poor.", 
                "According to Ledford, Northrop executives said they would build substantial parts of the bomber in Palmdale, creating about 1,500 jobs."]
    targets = ["Military experts say war is changing.", "Their eyes are very small, and they do not see well.",
                "According to Ledford, Northrop said they would build most of the bomber parts in Palmdale. It would create 1,500 jobs."]
    nltk.download('punkt')
    tokenized_sources = preprocess_texts(sources)
    tokenized_targets = preprocess_texts(targets)

    data = []
    example = namedtuple('example', 'ID, text_a, text_b, label')
    for i, (source, target) in enumerate(zip(sources, targets)):
        data.append(example(i, source, target, '0-0'))
    test_dataloader = create_Data_Loader(data_examples=data, args=args, set_type='test', batchsize=1, max_seq_length=128, tokenizer=tokenizer)

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(my_device) for t in batch)
        input_ids_a_and_b, input_ids_b_and_a, input_mask, segment_ids_a_and_b, segment_ids_b_and_a, sent1_valid_ids, sent2_valid_ids, sent1_wordpiece_length, sent2_wordpiece_length = batch
        with torch.no_grad():
            allign_id_pairs = model(input_ids_a_and_b=input_ids_a_and_b, input_ids_b_and_a=input_ids_b_and_a,
                                    attention_mask=input_mask, token_type_ids_a_and_b=segment_ids_a_and_b,
                                    token_type_ids_b_and_a=segment_ids_b_and_a,
                                    sent1_valid_ids=sent1_valid_ids, sent2_valid_ids=sent2_valid_ids,
                                    sent1_wordpiece_length=sent1_wordpiece_length,
                                    sent2_wordpiece_length=sent2_wordpiece_length)
        #print(list(allign_id_pairs))
        merged_sent2_allign_id_pairs = merge_sent2_ids(allign_id_pairs)
        merged_sent1_allign_id_pairs = merge_sent1_ids(merged_sent2_allign_id_pairs)
        allign_word_pairs = ids_to_words(merged_sent1_allign_id_pairs, tokenized_sources[step], tokenized_targets[step])
        print(merged_sent1_allign_id_pairs, allign_word_pairs)