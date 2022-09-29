from utils_extract_edit_operation import *
from utils_EditNTS import *
import tqdm
import random

def create_text_to_edit(edits, sent1_tok, sent2_tok, nlp):
    text_to_edit = []
    edits_ids = assign_ids_to_edits(edits, sent2_tok)

    doc_sent1 = nlp(" ".join(sent1_tok))
    sent1_tok_pos_lemma = []
    for i, token in enumerate(doc_sent1):
        sent1_tok_pos_lemma.append([token.text, token.pos_, token.lemma_])

    doc_sent2 = nlp(" ".join(sent2_tok))
    sent2_tok_pos_lemma = []
    for i, token in enumerate(doc_sent2):
        sent2_tok_pos_lemma.append([token.text, token.pos_, token.lemma_])
 
    for i in range(len(edits)):
        if edits[i] == 'DEL' or edits[i] == 'KEEP':
            text_to_edit.append([sent1_tok_pos_lemma[edits_ids[i]][0], sent1_tok_pos_lemma[edits_ids[i]][1], sent1_tok_pos_lemma[edits_ids[i]][2]])
        else:
            word_in_sent2 = sent2_tok_pos_lemma[edits_ids[i]][0]
            word_in_sent2 = 'ADD_' + word_in_sent2
            text_to_edit.append([word_in_sent2, sent2_tok_pos_lemma[edits_ids[i]][1], sent2_tok_pos_lemma[edits_ids[i]][2]])
    return text_to_edit

def get_edit_sequences(sent1_toks, sent2_toks, aligns):
    edit_sequences = []
    for i in tqdm.tqdm(range(len(aligns))):
        sent1_tok = sent1_toks[i]
        sent2_tok = sent2_toks[i]
        edits = sent2edit(sent1_tok, sent2_tok)
        #text_to_edit = create_text_to_edit(edits, sent1_tok, sent2_tok, nlp)
        ad_spans = extract_ad_spans(edits)
        d_spans = extract_d_spans(edits)
        a_spans = extract_a_spans(edits)

        splr_ids = extract_splr_ids(edits, ad_spans, sent2_tok)
        rep_ids, ad_ids = extract_rep_ad_ids(edits, ad_spans, sent2_tok, aligns[i], splr_ids)
        mvr_ids = extract_mvr_ids(edits, ad_spans, d_spans, a_spans, sent2_tok, aligns[i])
        d_ids = extract_d_ids(edits, d_spans, sent2_tok, aligns[i])
        a_ids = extract_a_ids(a_spans, mvr_ids)

        edit_sequence = splr_ids + rep_ids + ad_ids + mvr_ids + d_ids + a_ids
        edit_sequences.append(edit_sequence)
    return edit_sequences

def apply_edit_sequences(edit_sequences, sent1_toks, sent2_toks, nlp, max_cnt):
    random.seed(111)
    applied_sentences_all = []
    applied_edit_sequences_all = []

    for i in tqdm.tqdm(range(len(edit_sequences))):
        edit_sequence = edit_sequences[i]
        sent1_tok = sent1_toks[i]
        sent2_tok = sent2_toks[i]
        edits = sent2edit(sent1_tok, sent2_tok)
        text_to_edit = create_text_to_edit(edits, sent1_tok, sent2_tok, nlp)
        ad_spans = extract_ad_spans(edits)
        d_spans = extract_d_spans(edits)
        a_spans = extract_a_spans(edits)

        if len(edit_sequence) > 1:
            applied_sentences = []
            apply_sequences = []
            limit = max_cnt
            now_cnt = 0
            while limit != 0:
                if now_cnt == 2**(len(edit_sequence)) - 2:
                    break
                rn = random.randint(1, len(edit_sequence)-1)
                apply_sequence = random.sample(edit_sequence, rn)
                apply_sequence = [tuple(val) for val in apply_sequence]
                if set(apply_sequence) not in apply_sequences:
                    apply_sequences.append(set(apply_sequence))
                    now_cnt += 1
                    limit -= 1

                    apply_ad_spans = []
                    apply_d_spans = []
                    apply_a_spans = []
                    for apply_span in apply_sequence:
                        if apply_span[0] == 'splr_span' or apply_span[0] == 'rep_span' or apply_span[0] == 'ad_span':
                            ad_span_idx = apply_span[1]
                            ad_span = list(range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1))
                            apply_ad_spans += ad_span
                        if apply_span[0] == 'mvr_span':
                            for i in range(1, len(apply_span)):
                                if apply_span[i][0] == 'ad_span':
                                    ad_span_idx = apply_span[i][1]
                                    ad_span = list(range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1))
                                    apply_ad_spans += ad_span
                                if apply_span[i][0] == 'd_span':
                                    d_span_idx = apply_span[i][1]
                                    d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
                                    apply_d_spans += d_span
                                if apply_span[i][0] == 'a_span':
                                    a_span_idx = apply_span[i][1]
                                    a_span = list(range(a_spans[a_span_idx][0], a_spans[a_span_idx][1]+1))
                                    apply_a_spans += a_span
                        if apply_span[0] == 'd_span':
                            d_span_idx = apply_span[1]
                            d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
                            apply_d_spans += d_span
                        if apply_span[0] == 'a_span':
                            a_span_idx = apply_span[1]
                            a_span = list(range(a_spans[a_span_idx][0], a_spans[a_span_idx][1]+1))
                            apply_a_spans += a_span

                    applied_sentence_tok = []
                    for i in range(len(text_to_edit)):
                        if i in apply_ad_spans:
                            if edits[i] == 'DEL':
                                continue
                            else:
                                applied_sentence_tok.append(text_to_edit[i][0][4:])

                        elif i in apply_d_spans:
                            continue
                        elif i in apply_a_spans:
                            applied_sentence_tok.append(text_to_edit[i][0][4:])
                        else:
                            if edits[i] == 'DEL' or edits[i] == 'KEEP':
                                applied_sentence_tok.append(text_to_edit[i][0])
                            else:
                                continue 
                    applied_sentences.append(" ".join(applied_sentence_tok))
                    #applied_edit_sequences_all.append(apply_sequence)
            applied_sentences_all.append(applied_sentences)
            applied_edit_sequences_all.append(apply_sequences)
        else:
            applied_edit_sequences_all.append([])
            applied_sentences_all.append([])
    return applied_sentences_all, applied_edit_sequences_all


def apply_edit_sequences_sampling_without_replacement(edit_sequences, sent1_toks, sent2_toks, nlp):
    random.seed(111)
    applied_sentences_all = []
    applied_edit_sequences_all = []
    for i in tqdm.tqdm(range(len(edit_sequences))):
        edit_sequence = edit_sequences[i]
        sent1_tok = sent1_toks[i]
        sent2_tok = sent2_toks[i]
        edits = sent2edit(sent1_tok, sent2_tok)
        text_to_edit = create_text_to_edit(edits, sent1_tok, sent2_tok, nlp)
        ad_spans = extract_ad_spans(edits)
        d_spans = extract_d_spans(edits)
        a_spans = extract_a_spans(edits)

        if len(edit_sequence) > 1:
            applied_sentences = []
            apply_sequences = []
            while len(edit_sequence) !=0:
                rn = random.randint(1, max(len(edit_sequence)-1, 1))
                apply_sequence = random.sample(edit_sequence, rn)
                edit_sequence = [i for i in edit_sequence if i not in apply_sequence]
                apply_sequences.append(apply_sequence)
                apply_ad_spans = []
                apply_d_spans = []
                apply_a_spans = []
                for apply_span in apply_sequence:
                    if apply_span[0] == 'splr_span' or apply_span[0] == 'rep_span' or apply_span[0] == 'ad_span':
                        ad_span_idx = apply_span[1]
                        ad_span = list(range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1))
                        apply_ad_spans += ad_span
                    if apply_span[0] == 'mvr_span':
                        for i in range(1, len(apply_span)):
                            if apply_span[i][0] == 'ad_span':
                                ad_span_idx = apply_span[i][1]
                                ad_span = list(range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1))
                                apply_ad_spans += ad_span
                            if apply_span[i][0] == 'd_span':
                                d_span_idx = apply_span[i][1]
                                d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
                                apply_d_spans += d_span
                            if apply_span[i][0] == 'a_span':
                                a_span_idx = apply_span[i][1]
                                a_span = list(range(a_spans[a_span_idx][0], a_spans[a_span_idx][1]+1))
                                apply_a_spans += a_span
                    if apply_span[0] == 'd_span':
                        d_span_idx = apply_span[1]
                        d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
                        apply_d_spans += d_span
                    if apply_span[0] == 'a_span':
                        a_span_idx = apply_span[1]
                        a_span = list(range(a_spans[a_span_idx][0], a_spans[a_span_idx][1]+1))
                        apply_a_spans += a_span
                
                applied_sentence_tok = []
                for i in range(len(text_to_edit)):
                    if i in apply_ad_spans:
                        if edits[i] == 'DEL':
                            continue
                        else:
                            applied_sentence_tok.append(text_to_edit[i][0][4:])

                    elif i in apply_d_spans:
                        continue
                    elif i in apply_a_spans:
                        applied_sentence_tok.append(text_to_edit[i][0][4:])
                    else:
                        if edits[i] == 'DEL' or edits[i] == 'KEEP':
                            applied_sentence_tok.append(text_to_edit[i][0])
                        else:
                            continue 
                applied_sentences.append(" ".join(applied_sentence_tok))
                #applied_edit_sequences_all.append(apply_sequence)
            applied_sentences_all.append(applied_sentences)
            applied_edit_sequences_all.append(apply_sequences)
        else:
            applied_edit_sequences_all.append([])
            applied_sentences_all.append([])