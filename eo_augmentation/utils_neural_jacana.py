import argparse
from transformers import BertTokenizer
from collections import namedtuple
import random
import torch
import numpy as np
import re
import nltk
import pickle
import tqdm
from neural_jacana.model import *

class data_example:
    def __init__(self, ID, text_a, text_b, label):
        self.ID = ID
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def preprocess_texts(texts):
    tokenized_texts = []
    for text in texts:
        text = re.sub(r'[\(\)\`\'\"\:\;]', '', text)
        text = re.sub(r'-RRB-', '', text)
        text = re.sub(r'-LRB-', '', text)
        text = re.sub(r'DEL', 'del', text)
        text = re.sub(r'KEEP', 'keep', text)
        text = re.sub(r'ADD', 'add', text)
        tokenized_texts.append(nltk.word_tokenize(text))
    return tokenized_texts

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def check_inclusion(list1, list2):
    flag = False
    for i in list1:
        if i in list2:
            flag = True
    for i in list2:
        if i in list1:
            flag = True
    return flag

def merge_sent2_ids(align_id_pairs):
    '''
    merge sent2 ids aligned to sent1 ids.
    '''
    merged_sent2_align_id_pairs = []
    sorted_align_id_pairs = sorted(align_id_pairs, key=lambda x:(int(re.findall(r"\d+", x)[0]), int(re.findall(r"\d+", x)[1])))
    tuple_pairs = [(int(re.findall(r"\d+", i)[0]), int(re.findall(r"\d+", i)[1])) for i in sorted_align_id_pairs]
    for pair in tuple_pairs:
        keys = [i[0] for i in merged_sent2_align_id_pairs]
        keys_flatten = [x for row in keys for x in row]
        if pair[0] not in keys_flatten:
            merged_sent2_align_id_pairs.append(([pair[0]], [pair[1]]))
        else:
            ind_addval = [i for i in range(len(keys)) if pair[0] in keys[i]][0]
            merged_sent2_align_id_pairs[ind_addval][1].append(pair[1])
    return merged_sent2_align_id_pairs

def merge_sent1_ids(merged_sent2_align_id_pairs):
    '''
    merge sent1 ids having the same sent2 ids.
    '''
    merged_sent1_align_id_pairs = []
    dup_inds = []
    vals = [pair[1] for pair in merged_sent2_align_id_pairs]
    for pair in merged_sent2_align_id_pairs:
        dup_ind = [i for i, x in enumerate(vals) if x == pair[1]]
        if len(dup_ind) > 1:
            dup_inds.append(dup_ind)
    dup_inds = get_unique_list(dup_inds)

    if len(dup_inds) != 0: #if there are duplicate values in merged_sent2_align_id_pairs, they should be merged.
        keys_to_add = []
        for i in range(len(dup_inds)):
            key_to_add = []
            for j in range(len(merged_sent2_align_id_pairs)):
                if j in dup_inds[i]:
                    key_to_add.append(merged_sent2_align_id_pairs[j][0][0])
            if len(key_to_add) != 0:
                keys_to_add.append(key_to_add)
        
        pairs_to_add = []
        for i in range(len(dup_inds)):
            pairs_to_add.append((keys_to_add[i], merged_sent2_align_id_pairs[dup_inds[i][0]][1]))

        dup_inds_flatten = [x for row in dup_inds for x in row]
        for i in range(len(merged_sent2_align_id_pairs)):
            if i not in dup_inds_flatten:
                merged_sent1_align_id_pairs.append(merged_sent2_align_id_pairs[i])
        merged_sent1_align_id_pairs.extend(pairs_to_add)
        return merged_sent1_align_id_pairs
    
    else:
        return merged_sent2_align_id_pairs

def merge_align_ids_crossing(merged_ids):
    sent1_ids = [pair[0] for pair in merged_ids]
    sent2_ids = [pair[1] for pair in merged_ids]
    res = []
    added_sent1 = [0 for i in range(len(sent1_ids))]
    for i in range(len(sent1_ids)):
        sent2_ids_to_add = sent2_ids[i]
        sent1_correspond = sent1_ids[i]
        for j in range(i, len(sent1_ids)):
            if check_inclusion(sent1_ids[i], sent1_ids[j]) == True:
                if added_sent1[j] == 0 and i != j:
                    added_sent1[j] = 1
                    sent2_ids_to_add.extend(sent2_ids[j])
                    sent1_correspond.extend(sent1_ids[j])
                    sent1_correspond = get_unique_list(sent1_correspond)
        if len(sent2_ids_to_add) > 1 and added_sent1 == 0:
            res.append((sent1_correspond, sent2_ids_to_add))
            added_sent1[i] = 1
            #print(0)

    added_sent2 = [0 for i in range(len(sent2_ids))]
    for i in range(len(sent2_ids)):
        sent1_ids_to_add = sent1_ids[i]
        sent2_correspond = sent2_ids[i]
        for j in range(i, len(sent2_ids)):
            if check_inclusion(sent2_ids[i], sent2_ids[j]) == True:
                if added_sent2[j] == 0 and i != j:
                    added_sent2[j] = 1
                    sent1_ids_to_add.extend(sent1_ids[j])
                    sent2_correspond.extend(sent2_ids[j])
                    sent2_correspond = get_unique_list(sent2_correspond)
        if len(sent1_ids_to_add) > 1 and added_sent2[i] == 0:
            res.append((sent1_ids_to_add, sent2_correspond))
            added_sent2[i] = 1
            #print(1)

    for i in range(len(merged_ids)):
        if added_sent1[i] == 0 and added_sent2[i] == 0:
            res.append((sent1_ids[i], sent2_ids[i]))
            #print(2)
    #print(added_sent1, added_sent2)
    return res

def ids_to_words(merged_id_pairs, tokenized_sent1, tokenized_sent2):
    align_word_pairs = []
    for pair in merged_id_pairs:
        sent1_words = [tokenized_sent1[i] for i in pair[0]]
        sent2_words = [tokenized_sent2[i] for i in pair[1]]
        align_word_pairs.append((sent1_words, sent2_words))
    return align_word_pairs
    
def prepare_model(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = NeuralWordAligner(args)
    #my_device = torch.device('cpu')
    my_device = args.my_device
    model = model.to(my_device)

    checkpoint = torch.load('./neural_jacana/Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt', map_location=my_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_dataloader(sources, targets, args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    nltk.download('punkt')
    tokenized_sources = preprocess_texts(sources)
    tokenized_targets = preprocess_texts(targets)
    data = []
    for i, (tokenized_source, tokenized_target) in enumerate(zip(tokenized_sources, tokenized_targets)):
        data.append(data_example(i, ' '.join(tokenized_source), ' '.join(tokenized_target), '0-0'))
    test_dataloader = create_Data_Loader(data_examples=data, args=args, set_type='test', batchsize=args.batch_size, max_seq_length=128, tokenizer=tokenizer)
    with open('./src/test_dataloader.pickle', 'wb') as f:
        pickle.dump(test_dataloader, f)
    return test_dataloader

def get_alignment(model, args, sources, targets):
    #my_device = torch.device('cpu')
    my_device = args.my_device
    test_dataloader = get_dataloader(sources, targets, args)

    aligns_list = []
    for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        batch = tuple(t.to(my_device) for t in batch)
        input_ids_a_and_b, input_ids_b_and_a, input_mask, segment_ids_a_and_b, segment_ids_b_and_a, sent1_valid_ids, sent2_valid_ids, sent1_wordpiece_length, sent2_wordpiece_length = batch
        sent1_valid_ids = torch.tensor([normalize_1d_tensor_to_list(sent1_valid_ids[0])]).to(my_device)
        sent2_valid_ids = torch.tensor([normalize_1d_tensor_to_list(sent2_valid_ids[0])]).to(my_device)
        with torch.no_grad():
            decoded_results = model(input_ids_a_and_b=input_ids_a_and_b, input_ids_b_and_a=input_ids_b_and_a,
                                        attention_mask=input_mask, token_type_ids_a_and_b=segment_ids_a_and_b,
                                        token_type_ids_b_and_a=segment_ids_b_and_a,
                                        sent1_valid_ids=sent1_valid_ids, sent2_valid_ids=sent2_valid_ids,
                                        sent1_wordpiece_length=sent1_wordpiece_length,
                                        sent2_wordpiece_length=sent2_wordpiece_length)
        align_id_pairs = list(decoded_results[0])
        #print(align_id_pairs)
        merged_sent2_align_id_pairs = merge_sent2_ids(align_id_pairs)
        merged_sent1_align_id_pairs = merge_sent1_ids(merged_sent2_align_id_pairs)
        merged_id_pairs = merge_align_ids_crossing(merged_sent1_align_id_pairs)
        #align_word_pairs = ids_to_words(merged_id_pairs, tokenized_sources[step], tokenized_targets[step])
        aligns_list.append(merged_id_pairs)
    return aligns_list
